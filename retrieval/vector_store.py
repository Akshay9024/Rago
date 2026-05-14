from __future__ import annotations
import asyncio
import time
import uuid
from collections import deque
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
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams,
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
    """
    Per-session DRAM HNSW namespace with bounded capacity, duplicate suppression
    and FIFO eviction. hnswlib's add_items has no native dedup or capacity guard,
    so re-indexing the same chunk_id or exceeding max_elements would silently
    pollute the index or raise. This wrapper enforces both invariants explicitly.
    """

    def __init__(self, session_id: str, dim: int, max_vectors: int):
        self._session_id = session_id
        self._dim = dim
        self._max_vectors = max(1, max_vectors)
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=self._max_vectors,
            ef_construction=200,
            M=16,
            allow_replace_deleted=True,
        )
        self._index.set_ef(50)
        self._chunk_to_label: dict[str, int] = {}
        self._label_to_chunk: dict[int, str] = {}
        self._chunk_to_text: dict[str, str] = {}
        self._insertion_order: deque[int] = deque()
        self._next_label = 0
        self._deleted_pending: int = 0

    def __len__(self) -> int:
        return len(self._chunk_to_label)

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_to_label

    def _evict_oldest(self, n: int) -> int:
        evicted = 0
        while self._insertion_order and evicted < n:
            label = self._insertion_order.popleft()
            cid = self._label_to_chunk.pop(label, None)
            if cid is None:
                continue
            self._chunk_to_label.pop(cid, None)
            self._chunk_to_text.pop(cid, None)
            try:
                self._index.mark_deleted(label)
            except RuntimeError:
                continue
            evicted += 1
            self._deleted_pending += 1
        return evicted

    def add(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
    ) -> int:
        novel_ids: list[str] = []
        novel_texts: list[str] = []
        novel_vecs: list[np.ndarray] = []
        for cid, text, emb in zip(chunk_ids, texts, embeddings):
            if cid in self._chunk_to_label:
                continue
            novel_ids.append(cid)
            novel_texts.append(text)
            novel_vecs.append(emb)

        if not novel_ids:
            return 0

        overflow = (len(self._chunk_to_label) + len(novel_ids)) - self._max_vectors
        if overflow > 0:
            self._evict_oldest(overflow)

        labels: list[int] = []
        for _ in novel_ids:
            labels.append(self._next_label)
            self._next_label += 1

        arr = np.asarray(novel_vecs, dtype=np.float32)
        use_replace = self._deleted_pending > 0
        self._index.add_items(arr, labels, replace_deleted=use_replace)
        if use_replace:
            self._deleted_pending = max(0, self._deleted_pending - len(labels))

        for label, cid, text in zip(labels, novel_ids, novel_texts):
            self._chunk_to_label[cid] = label
            self._label_to_chunk[label] = cid
            self._chunk_to_text[cid] = text
            self._insertion_order.append(label)
        return len(novel_ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if len(self._chunk_to_label) == 0:
            return []
        k = min(top_k, len(self._chunk_to_label))
        labels, distances = self._index.knn_query(query.reshape(1, -1), k=k)
        results: list[RetrievalResult] = []
        for label, dist in zip(labels[0], distances[0]):
            cid = self._label_to_chunk.get(int(label))
            if cid is None:
                continue
            results.append(RetrievalResult(
                chunk_id=cid,
                text=self._chunk_to_text[cid],
                score=float(1.0 - dist),
                source="hot_tier",
            ))
        return results


class TieredVectorStore:
    def __init__(self, schema: RAGSchema, infra: InfraConfig, cache: RedisChunkCache):
        self._schema = schema
        self._infra = infra
        self._cache = cache
        self._cold: Optional[AsyncQdrantClient] = None
        self._hot: dict[str, HotTierNamespace] = {}

    async def initialize(self) -> None:
        if self._infra.qdrant_local_path:
            import os
            os.makedirs(self._infra.qdrant_local_path, exist_ok=True)
            self._cold = AsyncQdrantClient(path=self._infra.qdrant_local_path)
        else:
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
                    on_disk=True,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                    on_disk=True,
                ),
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20_000),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
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
        cold_soft_timeout_ms: int = 50,
        ef_full: int = 64,
        ef_soft: int = 16,
    ) -> list[RetrievalResult]:
        ns = self._hot_ns(session_id)
        hot_results = ns.search(query_embedding, top_k=top_k)

        if len(hot_results) >= top_k:
            return hot_results

        remaining = top_k - len(hot_results)
        cold_results = await self._search_cold_with_early_termination(
            query=query_embedding,
            top_k=remaining,
            hard_timeout_ms=cold_timeout_ms,
            soft_timeout_ms=cold_soft_timeout_ms,
            ef_full=ef_full,
            ef_soft=ef_soft,
        )

        hot_ids = {r.chunk_id for r in hot_results}
        novel_cold: list[RetrievalResult] = []
        for r in cold_results:
            if r.chunk_id in hot_ids or ns.contains(r.chunk_id):
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

    async def _search_cold_with_early_termination(
        self,
        query: np.ndarray,
        top_k: int,
        hard_timeout_ms: int,
        soft_timeout_ms: int,
        ef_full: int,
        ef_soft: int,
    ) -> list[RetrievalResult]:
        """
        Non-stall soft cancellation: race a wider-beam search against a soft deadline.
        If the full search misses the soft deadline, fall back to a narrower-beam query
        (lower hnsw_ef) bounded by the hard deadline, returning whatever the cheaper
        traversal accumulates. This is the production-safe analogue of an HNSW cancel:
        no thread is killed mid-traversal, the cheaper query runs to completion, and
        the index's beam-search state is never corrupted.
        """

        full_task = asyncio.create_task(
            self._issue_cold_query(query, top_k, ef_full, hard_timeout_ms)
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(full_task),
                timeout=soft_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            pass

        soft_task = asyncio.create_task(
            self._issue_cold_query(query, top_k, ef_soft, hard_timeout_ms)
        )
        done, _ = await asyncio.wait(
            {full_task, soft_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        winner = next(iter(done))
        for other in {full_task, soft_task} - done:
            other.cancel()
        try:
            return winner.result()
        except Exception:
            return []

    async def _issue_cold_query(
        self,
        query: np.ndarray,
        top_k: int,
        hnsw_ef: int,
        timeout_ms: int,
    ) -> list[RetrievalResult]:
        try:
            response = await asyncio.wait_for(
                self._cold.query_points(
                    collection_name=self._infra.qdrant_cold_collection,
                    query=query.tolist(),
                    limit=top_k,
                    search_params=SearchParams(
                        hnsw_ef=hnsw_ef,
                        exact=False,
                        quantization=QuantizationSearchParams(
                            ignore=False,
                            rescore=True,
                            oversampling=2.0,
                        ),
                    ),
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
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return []
        except Exception:
            return []

    async def close(self) -> None:
        if self._cold:
            await self._cold.close()
