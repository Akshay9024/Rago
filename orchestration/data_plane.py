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

    Late-response buffer: a response can arrive before the awaiting future is
    registered when the retrieval handler runs faster than the publish→suspend→
    wait_for_context path yields. Such responses are parked in `_buffered` with
    a TTL; when the matching future is eventually registered it resolves from
    the buffer immediately. Without this, fast in-process retrievals can drop
    responses and stall the decode worker until the inflight timeout fires.
    """

    def __init__(
        self,
        infra: InfraConfig,
        deployment: DeploymentMode,
        control_plane: NATSControlPlane,
        late_buffer_ttl_s: float = 60.0,
    ):
        self._infra = infra
        self._deployment = deployment
        self._control_plane = control_plane
        self._pending: dict[str, asyncio.Future[ContextPayload]] = {}
        self._buffered: dict[str, tuple[ContextPayload, float]] = {}
        self._buffer_ttl_s = float(late_buffer_ttl_s)
        self._lock = asyncio.Lock()
        self._subscriber_task: Optional[asyncio.Task] = None

    @staticmethod
    def _key(sequence_id: str, retrieval_round: int) -> str:
        return f"{sequence_id}:{retrieval_round}"

    def _gc_buffered_locked(self, now: float) -> None:
        expired = [k for k, (_, t) in self._buffered.items() if now - t > self._buffer_ttl_s]
        for k in expired:
            self._buffered.pop(k, None)

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
        future: Optional[asyncio.Future[ContextPayload]] = None
        async with self._lock:
            future = self._pending.pop(key, None)
            if future is None or future.done():
                now = asyncio.get_running_loop().time()
                self._gc_buffered_locked(now)
                self._buffered[key] = (payload, now)
                return
        if not future.done():
            future.set_result(payload)

    async def wait_for_context(
        self,
        sequence_id: str,
        retrieval_round: int,
        timeout: float = 30.0,
    ) -> ContextPayload:
        key = self._key(sequence_id, retrieval_round)
        future: asyncio.Future[ContextPayload] = asyncio.get_running_loop().create_future()

        async with self._lock:
            buffered = self._buffered.pop(key, None)
            if buffered is not None:
                future.set_result(buffered[0])
            else:
                self._pending[key] = future

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(key, None)
            raise

    async def expect_context(
        self,
        sequence_id: str,
        retrieval_round: int,
    ) -> None:
        key = self._key(sequence_id, retrieval_round)
        async with self._lock:
            if key in self._pending or key in self._buffered:
                return
            self._pending[key] = asyncio.get_running_loop().create_future()

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
