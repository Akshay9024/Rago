from __future__ import annotations
import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Awaitable, Optional


class SequencePriority(IntEnum):
    RESUMING = 0
    READY_CACHED = 1
    ACTIVE = 2
    WAITING = 3


@dataclass(order=True)
class _Entry:
    priority: int
    arrival: float
    seq_id: str = field(compare=False)
    cb: Callable[[], Awaitable[None]] = field(compare=False)


class PriorityNonStallScheduler:
    """
    Priority-aware non-stall scheduler implementing RAGO's scheduling principles.

    Key behaviors:
    - Sequences resuming after retrieval get highest scheduling priority so
      cached KV blocks are not evicted before the sequence resumes.
    - Sequences suspended for retrieval are tracked separately; the GPU is
      immediately yielded to other ready sequences (non-stall guarantee).
    - Concurrency cap enforces the RAGO decode batch size constraint.
    """

    def __init__(self, max_concurrent: int = 64):
        self._max = max_concurrent
        self._ready: list[_Entry] = []
        self._active: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        seq_id: str,
        priority: SequencePriority,
        cb: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._lock:
            heapq.heappush(self._ready, _Entry(
                priority=int(priority),
                arrival=time.perf_counter(),
                seq_id=seq_id,
                cb=cb,
            ))
            await self._drain()

    async def suspend(self, seq_id: str) -> None:
        async with self._lock:
            task = self._active.pop(seq_id, None)

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            await self._drain()

    async def resume(
        self,
        seq_id: str,
        cb: Callable[[], Awaitable[None]],
    ) -> None:
        async with self._lock:
            heapq.heappush(self._ready, _Entry(
                priority=int(SequencePriority.RESUMING),
                arrival=time.perf_counter(),
                seq_id=seq_id,
                cb=cb,
            ))
            await self._drain()

    async def _drain(self) -> None:
        while self._ready and len(self._active) < self._max:
            entry = heapq.heappop(self._ready)
            task = asyncio.create_task(
                self._guarded(entry),
                name=f"seq_{entry.seq_id}",
            )
            self._active[entry.seq_id] = task

    async def _guarded(self, entry: _Entry) -> None:
        try:
            await entry.cb()
        finally:
            async with self._lock:
                self._active.pop(entry.seq_id, None)
                await self._drain()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def queued_count(self) -> int:
        return len(self._ready)
