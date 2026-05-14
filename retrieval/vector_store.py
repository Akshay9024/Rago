from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

import hnswlib
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
    ScoredPoint,
)

from config.schema import RAGSchema
from config.settings import InfraConfig
from retrieval.cache import RedisChunkCache


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str


class HotTierNamespace:
    def __init__(self, session_id: str, dim: int, max_vectors: int):
        self._session_id = session_id
        self._dim = dim
        self._max_vectors = max_vectors
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=max_vectors, ef_construction=200, M=16)
        self._index.set_ef(50)
        self._label_to_chunk: dict[int, str] = {}
        self._chunk_to_text: dict[str, str] = {}
        self._count = 0

    def add(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
    ) -> None:
        labels = list(range(self._count, self._count + len(chunk_ids)))
        self._index.add_items(embeddings, labels)
        for label, cid, text in zip(labels, chunk_ids, texts):
            self._label_to_chunk[label] = cid
            self._chunk_to_text[cid] = text
        self._count += len(chunk_ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self._count == 0:
            return []
        k = min(top_k, self._count)
        labels, distances = self._index.knn_query(query.reshape(1, -1), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            cid = self._label_to_chunk[label]
            results.append(RetrievalResult(
                chunk_id=cid,
                text=self._chunk_to_text[cid],
                score=float(1.0 - dist),
                source="hot_tier",
            ))
        return results

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_to_text

    def __len__(self) -> int:
        return self._count


class TieredVectorStore:
    def __init__(self, schema: RAGSchema, infra: InfraConfig, cache: RedisChunkCache):
        self._schema = schema
        self._infra = infra
        self._cache = cache
        self._cold: Optional[AsyncQdrantClient] = None
        self._hot: dict[str, HotTierNamespace] = {}

    async def initialize(self) -> None:
        self._cold = AsyncQdrantClient(
            host=self._infra.qdrant_host,
            port=self._infra.qdrant_port,
            grpc_port=self._infra.qdrant_grpc_port,
            prefer_grpc=True,
            api_key=self._infra.qdrant_api_key,
        )
        existing = {
            c.name
            for c in (await self._cold.get_collections()).collections
        }
        if self._infra.qdrant_cold_collection not in existing:
            await self._cold.create_collection(
                collection_name=self._infra.qdrant_cold_collection,
                vectors_config=VectorParams(
                    size=self._schema.encoder_dim,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200, on_disk=True),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20_000),
            )

    def _hot_ns(self, session_id: str) -> HotTierNamespace:
        if session_id not in self._hot:
            self._hot[session_id] = HotTierNamespace(
                session_id=session_id,
                dim=self._schema.encoder_dim,
                max_vectors=self._schema.hot_tier_max_vectors,
            )
        return self._hot[session_id]

    def evict_session(self, session_id: str) -> None:
        self._hot.pop(session_id, None)

    async def index_documents(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        sources: list[str],
        session_id: Optional[str] = None,
    ) -> None:
        await self._cache.batch_set(chunk_ids=chunk_ids, texts=texts, embeddings=embeddings)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)),
                vector=emb.tolist(),
                payload={"chunk_id": cid, "text": text, "source": src},
            )
            for cid, emb, text, src in zip(chunk_ids, embeddings, texts, sources)
        ]
        await self._cold.upsert(
            collection_name=self._infra.qdrant_cold_collection,
            points=points,
        )

        if session_id:
            ns = self._hot_ns(session_id)
            ns.add(chunk_ids, embeddings, texts)

    async def search(
        self,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int = 5,
        cold_timeout_ms: int = 100,
    ) -> list[RetrievalResult]:
        ns = self._hot_ns(session_id)
        hot_results = ns.search(query_embedding, top_k=top_k)

        if len(hot_results) >= top_k:
            return hot_results

        remaining = top_k - len(hot_results)
        cold_results = await self._search_cold(query_embedding, remaining, cold_timeout_ms)

        hot_ids = {r.chunk_id for r in hot_results}
        novel_cold: list[RetrievalResult] = []
        for r in cold_results:
            if r.chunk_id in hot_ids:
                continue
            cached = await self._cache.get(r.chunk_id)
            if cached:
                r.text = cached["text"]
                chunk_emb = np.array(cached["embedding"], dtype=np.float32).reshape(1, -1)
            else:
                chunk_emb = None
            novel_cold.append(r)
            if r.text and chunk_emb is not None:
                ns.add(
                    chunk_ids=[r.chunk_id],
                    embeddings=chunk_emb,
                    texts=[r.text],
                )

        merged = hot_results + novel_cold
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    async def _search_cold(
        self,
        query: np.ndarray,
        top_k: int,
        timeout_ms: int,
    ) -> list[RetrievalResult]:
        try:
            response = await asyncio.wait_for(
                self._cold.query_points(
                    collection_name=self._infra.qdrant_cold_collection,
                    query=query,
                    limit=top_k,
                    search_params=SearchParams(hnsw_ef=64, exact=False),
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=timeout_ms / 1000.0,
            )
            return [
                RetrievalResult(
                    chunk_id=p.payload.get("chunk_id", str(p.id)),
                    text=p.payload.get("text", ""),
                    score=float(p.score),
                    source="cold_tier",
                )
                for p in response.points
            ]
        except asyncio.TimeoutError:
            return []

    async def close(self) -> None:
        if self._cold:
            await self._cold.close()
