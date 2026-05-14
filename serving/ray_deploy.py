from __future__ import annotations
import asyncio
import os
from typing import Optional

import ray
from ray import serve

from config.schema import RAGSchema
from config.settings import SystemConfig
from pipeline.iterative_rag import IterativeRAGPipeline, IterativeRAGRequest


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    max_ongoing_requests=64,
    health_check_period_s=15,
    health_check_timeout_s=60,
    graceful_shutdown_timeout_s=30,
)
class IterativeRAGActor:
    def __init__(self, schema_overrides: dict, config_profile: str):
        self._schema = RAGSchema(**schema_overrides)
        self._config_profile = config_profile
        self._pipeline: Optional[IterativeRAGPipeline] = None
        self._init_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        async with self._init_lock:
            if self._pipeline is not None:
                return
            config = self._resolve_config(self._config_profile)
            self._pipeline = await IterativeRAGPipeline.create(self._schema, config)

    @staticmethod
    def _resolve_config(profile: str) -> SystemConfig:
        if profile == "rtx4090_local":
            return SystemConfig.for_rtx4090_local()
        elif profile == "a6000_local":
            return SystemConfig.for_a6000_local()
        elif profile == "a6000":
            return SystemConfig.for_a6000_ssh(os.environ["REMOTE_HOST"])
        elif profile == "a100":
            return SystemConfig.for_a100_ssh(os.environ["REMOTE_HOST"])
        elif profile == "a100_80g":
            return SystemConfig.for_a100_80g_ssh(os.environ["REMOTE_HOST"])
        elif profile == "h100":
            return SystemConfig.for_h100_ssh(os.environ["REMOTE_HOST"])
        elif profile == "io_net":
            host = os.environ["REMOTE_HOST"]
            world = int(os.getenv("WORLD_SIZE", "4"))
            return SystemConfig.for_io_net_cluster(host, world)
        raise ValueError(f"Unknown profile: {profile}")

    async def generate(self, query: str, session_id: str) -> dict:
        await self._ensure_ready()
        req = IterativeRAGRequest(query=query, session_id=session_id)
        resp = await self._pipeline.generate(req)
        return {
            "request_id": resp.request_id,
            "generated_text": resp.generated_text,
            "retrieval_count": resp.retrieval_count,
            "total_latency_ms": resp.total_latency_ms,
            "retrieval_latencies_ms": resp.retrieval_latencies_ms,
        }

    async def ingest(self, texts: list[str], sources: list[str], session_id: Optional[str] = None) -> dict:
        await self._ensure_ready()
        n = await self._pipeline.ingest(texts=texts, sources=sources, session_id=session_id)
        return {"indexed_chunks": n, "status": "ok"}

    async def reconfigure(self, config: dict) -> None:
        pass

    def check_health(self) -> None:
        pass


def deploy(
    schema_overrides: Optional[dict] = None,
    config_profile: str = "rtx4090_local",
    ray_address: Optional[str] = None,
    http_host: str = "0.0.0.0",
    http_port: int = 8000,
) -> serve.Application:
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    serve.start(
        http_options={"host": http_host, "port": http_port},
        dedicated_cpu=True,
    )

    app = IterativeRAGActor.bind(
        schema_overrides=schema_overrides or {},
        config_profile=config_profile,
    )
    serve.run(app, name="iterative-rag", route_prefix="/")
    return app
