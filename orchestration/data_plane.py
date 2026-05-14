from __future__ import annotations
import asyncio
from dataclasses import dataclass

from config.settings import InfraConfig, DeploymentMode


@dataclass
class ContextPayload:
    sequence_id: str
    retrieval_round: int
    passages: list[str]
    retrieval_latency_ms: float


class DataPlane:
    """
    Context delivery plane bridging the retrieval worker and the decode worker.

    Keyed by (sequence_id, retrieval_round) to guarantee idempotency: a stale
    or duplicate delivery from a prior round cannot resume the wrong generation
    step, even if NATS at-least-once redelivers after a worker restart.
    """

    def __init__(self, infra: InfraConfig, deployment: DeploymentMode):
        self._infra = infra
        self._deployment = deployment
        self._pending: dict[str, asyncio.Future[ContextPayload]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(sequence_id: str, retrieval_round: int) -> str:
        return f"{sequence_id}:{retrieval_round}"

    async def initialize(self) -> None:
        pass

    async def wait_for_context(
        self,
        sequence_id: str,
        retrieval_round: int,
        timeout: float = 30.0,
    ) -> ContextPayload:
        key = self._key(sequence_id, retrieval_round)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[ContextPayload] = loop.create_future()

        async with self._lock:
            self._pending[key] = future

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(key, None)
            raise

    async def deliver_context(self, payload: ContextPayload) -> None:
        key = self._key(payload.sequence_id, payload.retrieval_round)
        async with self._lock:
            future = self._pending.pop(key, None)
        if future and not future.done():
            future.set_result(payload)

    async def close(self) -> None:
        async with self._lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
