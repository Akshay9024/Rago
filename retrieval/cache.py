from __future__ import annotations
import json
import numpy as np
import redis.asyncio as aioredis
from typing import Optional
from config.settings import InfraConfig


class RedisChunkCache:
    _KEY_PREFIX = "chunk:"

    def __init__(self, infra: InfraConfig):
        self._infra = infra
        self._client: Optional[aioredis.Redis] = None

    async def initialize(self) -> None:
        self._client = aioredis.Redis(
            host=self._infra.redis_host,
            port=self._infra.redis_port,
            db=self._infra.redis_db,
            decode_responses=False,
            max_connections=100,
            socket_keepalive=True,
        )
        await self._client.ping()
        await self._client.config_set(
            "maxmemory", f"{self._infra.redis_max_memory_gb}gb"
        )
        await self._client.config_set("maxmemory-policy", "allkeys-lru")

    async def get(self, chunk_id: str) -> Optional[dict]:
        raw = await self._client.get(f"{self._KEY_PREFIX}{chunk_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def get_text(self, chunk_id: str) -> Optional[str]:
        entry = await self.get(chunk_id)
        return entry["text"] if entry else None

    async def set(self, chunk_id: str, text: str, embedding: np.ndarray) -> None:
        payload = json.dumps({
            "text": text,
            "embedding": embedding.astype(np.float16).tolist(),
        })
        await self._client.set(f"{self._KEY_PREFIX}{chunk_id}", payload)

    async def batch_set(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
    ) -> None:
        pipe = self._client.pipeline(transaction=False)
        for cid, text, emb in zip(chunk_ids, texts, embeddings):
            payload = json.dumps({
                "text": text,
                "embedding": emb.astype(np.float16).tolist(),
            })
            pipe.set(f"{self._KEY_PREFIX}{cid}", payload)
        await pipe.execute()

    async def exists(self, chunk_id: str) -> bool:
        return bool(await self._client.exists(f"{self._KEY_PREFIX}{chunk_id}"))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
