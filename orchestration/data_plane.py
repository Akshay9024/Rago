from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Optional

from config.settings import InfraConfig, DeploymentMode
from orchestration.nats_client import NATSControlPlane, RetrievalResponse


@dataclass
class ContextPayload:
    sequence_id: str
    retrieval_round: int
    passages: list[str]
    retrieval_latency_ms: float


class DataPlane:
    """
    Context delivery plane bridging retrieval workers and the decode worker.

    Responses are routed back from workers via the NATSControlPlane's per-instance
    reply subject, so workers may live in this process (single-node dev) or on
    separate retrieval nodes (multi-node prod) without changing call sites.

    The local future map is keyed by (sequence_id, retrieval_round) for idempotency:
    duplicate redeliveries from at-least-once transport cannot resume the wrong
    generation step, and late responses for a timed-out future are silently dropped.
    """

    def __init__(
        self,
        infra: InfraConfig,
        deployment: DeploymentMode,
        control_plane: NATSControlPlane,
    ):
        self._infra = infra
        self._deployment = deployment
        self._control_plane = control_plane
        self._pending: dict[str, asyncio.Future[ContextPayload]] = {}
        self._lock = asyncio.Lock()
        self._subscriber_task: Optional[asyncio.Task] = None

    @staticmethod
    def _key(sequence_id: str, retrieval_round: int) -> str:
        return f"{sequence_id}:{retrieval_round}"

    async def initialize(self) -> None:
        self._subscriber_task = asyncio.create_task(
            self._control_plane.subscribe_retrieval_responses(
                self._on_response,
            ),
            name="data_plane_response_subscriber",
        )

    async def _on_response(self, resp: RetrievalResponse) -> None:
        payload = ContextPayload(
            sequence_id=resp.sequence_id,
            retrieval_round=resp.retrieval_round,
            passages=resp.passages,
            retrieval_latency_ms=resp.latency_ms,
        )
        key = self._key(resp.sequence_id, resp.retrieval_round)
        async with self._lock:
            future = self._pending.pop(key, None)
        if future and not future.done():
            future.set_result(payload)

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

    async def publish_context(
        self,
        reply_subject: str,
        payload: ContextPayload,
    ) -> None:
        resp = RetrievalResponse(
            sequence_id=payload.sequence_id,
            retrieval_round=payload.retrieval_round,
            passages=payload.passages,
            latency_ms=payload.retrieval_latency_ms,
        )
        await self._control_plane.publish_retrieval_response(reply_subject, resp)

    async def close(self) -> None:
        if self._subscriber_task is not None:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except (asyncio.CancelledError, Exception):
                pass
            self._subscriber_task = None
        async with self._lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
