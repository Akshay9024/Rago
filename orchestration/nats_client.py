from __future__ import annotations
import asyncio
import json
import uuid
from dataclasses import dataclass, asdict, field
from typing import Callable, Awaitable, Optional

from config.settings import InfraConfig


@dataclass
class RetrievalRequest:
    sequence_id: str
    retrieval_round: int
    query_text: str
    session_id: str
    trace_id: str
    reply_subject: str = ""

    @property
    def idempotency_key(self) -> str:
        return f"{self.sequence_id}:{self.retrieval_round}"

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "RetrievalRequest":
        return cls(**json.loads(data))


@dataclass
class RetrievalResponse:
    sequence_id: str
    retrieval_round: int
    passages: list[str]
    latency_ms: float

    @property
    def correlation_key(self) -> str:
        return f"{self.sequence_id}:{self.retrieval_round}"

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "RetrievalResponse":
        return cls(**json.loads(data))


class _LocalBus:
    """
    Single-process broadcast bus: pub/sub by subject, with a TTL'd dedup window
    on the request stream that mirrors JetStream's exactly-once semantics.
    """

    def __init__(self, dedup_window_seconds: int) -> None:
        self._request_queue: asyncio.Queue[RetrievalRequest] = asyncio.Queue()
        self._response_queues: dict[str, asyncio.Queue[RetrievalResponse]] = {}
        self._seen: dict[str, float] = {}
        self._dedup_window = float(dedup_window_seconds)
        self._lock = asyncio.Lock()

    async def publish_request(self, req: RetrievalRequest) -> None:
        now = asyncio.get_running_loop().time()
        async with self._lock:
            for k, t in list(self._seen.items()):
                if now - t > self._dedup_window:
                    self._seen.pop(k, None)
            if req.idempotency_key in self._seen:
                return
            self._seen[req.idempotency_key] = now
        await self._request_queue.put(req)

    async def subscribe_requests(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        while True:
            req = await self._request_queue.get()
            try:
                await handler(req)
            except Exception:
                pass
            finally:
                self._request_queue.task_done()

    async def publish_response(self, subject: str, resp: RetrievalResponse) -> None:
        async with self._lock:
            q = self._response_queues.get(subject)
        if q is None:
            return
        await q.put(resp)

    async def subscribe_responses(
        self,
        subject: str,
        handler: Callable[[RetrievalResponse], Awaitable[None]],
    ) -> None:
        async with self._lock:
            q = self._response_queues.setdefault(subject, asyncio.Queue())
        while True:
            resp = await q.get()
            try:
                await handler(resp)
            except Exception:
                pass
            finally:
                q.task_done()


class _NATSBus:
    """
    Real NATS transport. Requests flow through a JetStream durable stream so
    retries survive worker restarts and the dedup window enforces exactly-once
    delivery on the consumer side. Responses flow through plain core-NATS
    subjects: they are short-lived, latency-sensitive, and re-issuing a
    retrieval is cheaper than persisting every response payload.
    """

    def __init__(self, infra: InfraConfig) -> None:
        self._infra = infra
        self._nc = None
        self._js = None

    @property
    def _req_subject(self) -> str:
        return f"{self._infra.nats_stream}.retrieval_request"

    async def initialize(self) -> None:
        import nats
        from nats.js.api import StreamConfig

        self._nc = await nats.connect(
            servers=[self._infra.nats_url],
            max_reconnect_attempts=10,
            reconnect_time_wait=2,
        )
        self._js = self._nc.jetstream()
        try:
            await self._js.stream_info(self._infra.nats_stream)
        except Exception:
            await self._js.add_stream(StreamConfig(
                name=self._infra.nats_stream,
                subjects=[self._req_subject],
                max_msgs=100_000,
                duplicate_window=self._infra.nats_dedup_window_seconds,
            ))

    async def publish_request(self, req: RetrievalRequest) -> None:
        await self._js.publish(
            self._req_subject,
            req.to_json(),
            headers={"Nats-Msg-Id": req.idempotency_key},
        )

    async def subscribe_requests(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        from nats.js.api import ConsumerConfig, AckPolicy, DeliverPolicy

        sub = await self._js.pull_subscribe(
            subject=self._req_subject,
            durable=self._infra.nats_consumer_name,
            config=ConsumerConfig(
                ack_policy=AckPolicy.EXPLICIT,
                deliver_policy=DeliverPolicy.ALL,
                max_deliver=3,
            ),
        )
        while True:
            try:
                msgs = await sub.fetch(batch=32, timeout=1.0)
                for msg in msgs:
                    req = RetrievalRequest.from_json(msg.data)
                    try:
                        await handler(req)
                        await msg.ack()
                    except Exception:
                        await msg.nak()
            except Exception:
                await asyncio.sleep(0.1)

    async def publish_response(self, subject: str, resp: RetrievalResponse) -> None:
        await self._nc.publish(subject, resp.to_json())

    async def subscribe_responses(
        self,
        subject: str,
        handler: Callable[[RetrievalResponse], Awaitable[None]],
    ) -> None:
        async def _on_msg(msg) -> None:
            try:
                resp = RetrievalResponse.from_json(msg.data)
                await handler(resp)
            except Exception:
                pass

        await self._nc.subscribe(subject, cb=_on_msg)
        stop = asyncio.Event()
        await stop.wait()

    async def close(self) -> None:
        if self._nc:
            await self._nc.close()


class NATSControlPlane:
    def __init__(self, infra: InfraConfig):
        self._infra = infra
        self._use_local = infra.nats_url == "local"
        self._bus: Optional[_LocalBus | _NATSBus] = None
        self._instance_id = uuid.uuid4().hex[:12]

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def response_subject(self) -> str:
        return f"{self._infra.nats_stream}.retrieval_response.{self._instance_id}"

    async def initialize(self) -> None:
        if self._use_local:
            self._bus = _LocalBus(self._infra.nats_dedup_window_seconds)
        else:
            bus = _NATSBus(self._infra)
            await bus.initialize()
            self._bus = bus

    async def publish_retrieval_request(self, req: RetrievalRequest) -> None:
        if not req.reply_subject:
            req.reply_subject = self.response_subject
        await self._bus.publish_request(req)

    async def subscribe_retrieval_requests(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        await self._bus.subscribe_requests(handler)

    async def publish_retrieval_response(
        self,
        reply_subject: str,
        resp: RetrievalResponse,
    ) -> None:
        await self._bus.publish_response(reply_subject, resp)

    async def subscribe_retrieval_responses(
        self,
        handler: Callable[[RetrievalResponse], Awaitable[None]],
        subject: Optional[str] = None,
    ) -> None:
        await self._bus.subscribe_responses(subject or self.response_subject, handler)

    async def close(self) -> None:
        if isinstance(self._bus, _NATSBus):
            await self._bus.close()
