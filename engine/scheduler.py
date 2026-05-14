from __future__ import annotations
import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class SequencePriority(IntEnum):
    RESUMING = 0
    READY_CACHED = 1
    ACTIVE = 2
    WAITING = 3


@dataclass(order=True)
class _Waiter:
    priority: int
    arrival: float
    seq: int
    seq_id: str = field(compare=False)
    event: asyncio.Event = field(compare=False)


class PriorityNonStallScheduler:
    """
    Cooperative coordination primitive realising RAGO priority-aware non-stall scheduling.

    Concurrency is capped at max_concurrent. acquire() blocks until a slot is granted,
    with the priority heap ensuring RESUMING > READY_CACHED > ACTIVE > WAITING. suspend()
    releases the slot back to the pool (yielding to other ready sequences while the caller
    waits for retrieval); resume() re-acquires at RESUMING priority so the suspended
    sequence outranks fresh admissions.
    """

    def __init__(self, max_concurrent: int = 64):
        self._max = max_concurrent
        self._active: set[str] = set()
        self._suspended: set[str] = set()
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._lock = asyncio.Lock()

    async def acquire(self, seq_id: str, priority: SequencePriority) -> None:
        event: Optional[asyncio.Event] = None
        async with self._lock:
            if len(self._active) < self._max:
                self._active.add(seq_id)
                return
            event = asyncio.Event()
            heapq.heappush(self._waiters, _Waiter(
                priority=int(priority),
                arrival=time.perf_counter(),
                seq=next(self._seq),
                seq_id=seq_id,
                event=event,
            ))
        await event.wait()

    async def release(self, seq_id: str) -> None:
        async with self._lock:
            self._active.discard(seq_id)
            self._suspended.discard(seq_id)
            self._wake_next_locked()

    async def suspend(self, seq_id: str) -> None:
        async with self._lock:
            if seq_id in self._active:
                self._active.discard(seq_id)
                self._suspended.add(seq_id)
                self._wake_next_locked()

    async def resume(self, seq_id: str) -> None:
        event: Optional[asyncio.Event] = None
        async with self._lock:
            self._suspended.discard(seq_id)
            if len(self._active) < self._max:
                self._active.add(seq_id)
                return
            event = asyncio.Event()
            heapq.heappush(self._waiters, _Waiter(
                priority=int(SequencePriority.RESUMING),
                arrival=time.perf_counter(),
                seq=next(self._seq),
                seq_id=seq_id,
                event=event,
            ))
        await event.wait()

    def _wake_next_locked(self) -> None:
        while self._waiters and len(self._active) < self._max:
            w = heapq.heappop(self._waiters)
            self._active.add(w.seq_id)
            w.event.set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def suspended_count(self) -> int:
        return len(self._suspended)

    @property
    def queued_count(self) -> int:
        return len(self._waiters)
