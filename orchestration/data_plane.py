from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional

from config.settings import InfraConfig, DeploymentMode


@dataclass
class ContextPayload:
    sequence_id: str
    retrieval_round: int
    passages: list[str]
    retrieval_latency_ms: float


class DataPlane:
    """
    Context delivery plane: bridges the retrieval worker and the decode worker.

    Single-node mode uses asyncio Futures for zero-copy local delivery.
    Multi-node mode (future) would replace this with a gRPC streaming channel
    that transfers token IDs directly to the target GPU memory address.
    """

    def __init__(self, infra: InfraConfig, deployment: DeploymentMode):
        self._infra = infra
        self._deployment = deployment
        self._pending: dict[str, asyncio.Future[ContextPayload]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def wait_for_context(
        self,
        sequence_id: str,
        timeout: float = 30.0,
    ) -> ContextPayload:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[ContextPayload] = loop.create_future()

        async with self._lock:
            self._pending[sequence_id] = future

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(sequence_id, None)
            raise

    async def deliver_context(self, payload: ContextPayload) -> None:
        async with self._lock:
            future = self._pending.pop(payload.sequence_id, None)
        if future and not future.done():
            future.set_result(payload)

    async def close(self) -> None:
        async with self._lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
