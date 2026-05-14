from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional

from config.settings import InfraConfig


@dataclass
class RetrievalRequest:
    sequence_id: str
    retrieval_round: int
    query_text: str
    session_id: str
    trace_id: str

    @property
    def idempotency_key(self) -> str:
        return f"{self.sequence_id}:{self.retrieval_round}"

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "RetrievalRequest":
        return cls(**json.loads(data))


@dataclass
class RetrievalResponse:
    sequence_id: str
    retrieval_round: int
    passages: list[str]
    latency_ms: float

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "RetrievalResponse":
        return cls(**json.loads(data))


class _LocalBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[RetrievalRequest] = asyncio.Queue()
        self._seen: set[str] = set()

    async def publish(self, req: RetrievalRequest) -> None:
        key = req.idempotency_key
        if key in self._seen:
            return
        self._seen.add(key)
        await self._queue.put(req)

    async def subscribe(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        while True:
            req = await self._queue.get()
            try:
                await handler(req)
            except Exception:
                pass
            finally:
                self._queue.task_done()


class _NATSBus:
    def __init__(self, infra: InfraConfig) -> None:
        self._infra = infra
        self._nc = None
        self._js = None

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
            await self._js.find_stream(self._infra.nats_stream)
        except Exception:
            await self._js.add_stream(StreamConfig(
                name=self._infra.nats_stream,
                subjects=[
                    f"{self._infra.nats_stream}.retrieval_request",
                ],
                max_msgs=100_000,
                duplicate_window=self._infra.nats_dedup_window_seconds * 10 ** 9,
            ))

    async def publish(self, req: RetrievalRequest) -> None:
        await self._js.publish(
            f"{self._infra.nats_stream}.retrieval_request",
            req.to_json(),
            headers={"Nats-Msg-Id": req.idempotency_key},
        )

    async def subscribe(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        from nats.js.api import ConsumerConfig, AckPolicy, DeliverPolicy

        sub = await self._js.pull_subscribe(
            subject=f"{self._infra.nats_stream}.retrieval_request",
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

    async def close(self) -> None:
        if self._nc:
            await self._nc.close()


class NATSControlPlane:
    def __init__(self, infra: InfraConfig):
        self._infra = infra
        self._use_local = infra.nats_url == "local"
        self._bus: Optional[_LocalBus | _NATSBus] = None

    async def initialize(self) -> None:
        if self._use_local:
            self._bus = _LocalBus()
        else:
            bus = _NATSBus(self._infra)
            await bus.initialize()
            self._bus = bus

    async def publish_retrieval_request(self, req: RetrievalRequest) -> None:
        await self._bus.publish(req)

    async def subscribe_retrieval_requests(
        self,
        handler: Callable[[RetrievalRequest], Awaitable[None]],
    ) -> None:
        await self._bus.subscribe(handler)

    async def close(self) -> None:
        if isinstance(self._bus, _NATSBus):
            await self._bus.close()
