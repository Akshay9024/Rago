from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class StopRagTransition:
    state: "StopRagState"
    action: "Action"
    reward: float
    next_state: Optional["StopRagState"]


class Action(IntEnum):
    STOP = 0
    CONTINUE = 1


@dataclass(frozen=True)
class StopRagState:
    retrieval_count: int
    entropy_bucket: int
    coverage_bucket: int

    def index(self, max_retrievals: int) -> tuple[int, int, int]:
        return (
            min(self.retrieval_count, max_retrievals - 1),
            self.entropy_bucket,
            self.coverage_bucket,
        )


class StopRagController:
    """
    Offline Q-learning MDP stop controller.

    Models the iterative process as a finite-horizon MDP where the agent
    decides after each retrieval whether to fetch again (CONTINUE) or halt
    (STOP). The Q-table is seeded with hand-crafted priors and can be
    refined via online TD updates from production traces.

    State: (retrieval_count, entropy_bucket, coverage_bucket)
    Action: {STOP=0, CONTINUE=1}
    Reward: quality_proxy - lambda * latency_cost
    """

    def __init__(
        self,
        max_retrievals: int = 8,
        lambda_cost: float = 0.1,
        q_table_path: Optional[str] = None,
    ):
        self._max = max_retrievals
        self._lambda = lambda_cost
        self._q = self._init_q()
        if q_table_path and Path(q_table_path).exists():
            self._q = np.load(q_table_path)

    def _init_q(self) -> np.ndarray:
        q = np.zeros((self._max, 3, 2, 2), dtype=np.float32)
        for r in range(self._max):
            decay = max(0.0, 1.0 - r / self._max)
            q[r, 2, :, Action.CONTINUE] = 2.0 * decay
            q[r, 2, :, Action.STOP] = 0.5
            q[r, 1, :, Action.CONTINUE] = 1.2 * decay
            q[r, 1, :, Action.STOP] = 0.9
            q[r, 0, :, Action.STOP] = 1.5
            q[r, 0, :, Action.CONTINUE] = 0.2 * decay
            q[r, :, 1, Action.STOP] += 0.6
            q[r, :, 1, Action.CONTINUE] -= 0.3
        return q

    @staticmethod
    def _entropy_bucket(entropy: float) -> int:
        if entropy < 1.5:
            return 0
        if entropy < 2.5:
            return 1
        return 2

    @staticmethod
    def _coverage_bucket(context_tokens: int, max_context: int) -> int:
        return 1 if context_tokens / max(1, max_context) > 0.6 else 0

    def should_stop(
        self,
        retrieval_count: int,
        mean_entropy: float,
        context_tokens: int,
        max_context_tokens: int = 8192,
    ) -> bool:
        if retrieval_count >= self._max:
            return True
        if retrieval_count == 0:
            return False

        state = StopRagState(
            retrieval_count=retrieval_count,
            entropy_bucket=self._entropy_bucket(mean_entropy),
            coverage_bucket=self._coverage_bucket(context_tokens, max_context_tokens),
        )
        ri, ei, ci = state.index(self._max)
        q_stop = float(self._q[ri, ei, ci, Action.STOP])
        q_cont = float(self._q[ri, ei, ci, Action.CONTINUE]) - self._lambda * retrieval_count
        return q_stop > q_cont

    def update(
        self,
        state: StopRagState,
        action: Action,
        reward: float,
        next_state: Optional[StopRagState],
        alpha: float = 0.05,
        gamma: float = 0.9,
    ) -> None:
        ri, ei, ci = state.index(self._max)
        if next_state is None:
            td_target = reward
        else:
            nri, nei, nci = next_state.index(self._max)
            td_target = reward + gamma * float(np.max(self._q[nri, nei, nci]))
        self._q[ri, ei, ci, action] += alpha * (td_target - self._q[ri, ei, ci, action])

    def train_offline(
        self,
        transitions: Iterable[StopRagTransition],
        alpha: float,
        gamma: float,
        epochs: int = 1,
    ) -> None:
        batch = list(transitions)
        if not batch:
            return
        for _ in range(max(1, epochs)):
            for t in batch:
                self.update(
                    state=t.state,
                    action=t.action,
                    reward=t.reward,
                    next_state=t.next_state,
                    alpha=alpha,
                    gamma=gamma,
                )

    @classmethod
    def state_from_trace(
        cls,
        retrieval_count: int,
        mean_entropy: float,
        context_tokens: int,
        max_context_tokens: int,
        max_retrievals: int,
    ) -> StopRagState:
        return StopRagState(
            retrieval_count=min(retrieval_count, max_retrievals - 1),
            entropy_bucket=cls._entropy_bucket(mean_entropy),
            coverage_bucket=cls._coverage_bucket(context_tokens, max_context_tokens),
        )

    def save(self, path: str) -> None:
        np.save(path, self._q)
