from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class IntrygueState:
    suppressed_until_token: int = 0
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=32))

    def reset_entropy(self) -> None:
        self.entropy_history.clear()


class IntrygueController:
    """
    Hybrid induction-aware entropy gating (INTRYGUE).

    Standard entropy conflates model uncertainty with the mechanistic side-effect
    of induction heads copying grounded facts, producing false retrieval triggers.
    The copy score acts as a lightweight proxy for induction head activity: high
    n-gram overlap between recent generation and retrieved context indicates the
    model is copying confidently, so the entropy signal is suppressed.
    """

    def __init__(
        self,
        entropy_threshold: float = 2.5,
        copy_threshold: float = 0.3,
        window: int = 8,
        ngram_size: int = 3,
        suppression_window: int = 32,
        warmup_tokens: int = 4,
    ):
        self._e_thresh = entropy_threshold
        self._c_thresh = copy_threshold
        self._window = window
        self._ngram = ngram_size
        self._sup_window = suppression_window
        self._warmup = warmup_tokens

    def compute_entropy(self, logprobs: dict[int, float]) -> float:
        if not logprobs:
            return 0.0
        log_p = np.array(list(logprobs.values()), dtype=np.float64)
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def compute_copy_score(
        self,
        generated_ids: list[int],
        context_ids: list[int],
    ) -> float:
        if len(generated_ids) < self._ngram or not context_ids:
            return 0.0
        recent = generated_ids[-self._window:]
        context_ngrams = {
            tuple(context_ids[i: i + self._ngram])
            for i in range(len(context_ids) - self._ngram + 1)
        }
        possible = max(1, len(recent) - self._ngram + 1)
        matches = sum(
            1
            for i in range(possible)
            if tuple(recent[i: i + self._ngram]) in context_ngrams
        )
        return matches / possible

    def should_retrieve(
        self,
        logprobs: dict[int, float],
        generated_ids: list[int],
        context_ids: list[int],
        current_token_index: int,
        state: IntrygueState,
    ) -> tuple[bool, IntrygueState]:
        if current_token_index < state.suppressed_until_token:
            return False, state

        entropy = self.compute_entropy(logprobs)
        state.entropy_history.append(entropy)

        if len(state.entropy_history) < self._warmup:
            return False, state

        avg_entropy = float(np.mean(state.entropy_history))
        copy_score = self.compute_copy_score(generated_ids, context_ids)

        triggered = avg_entropy > self._e_thresh and copy_score < self._c_thresh

        if triggered:
            new_state = IntrygueState(
                suppressed_until_token=current_token_index + self._sup_window,
            )
            return True, new_state

        return False, state

    def on_context_injected(
        self,
        current_token_index: int,
        state: IntrygueState,
    ) -> IntrygueState:
        return IntrygueState(
            suppressed_until_token=current_token_index + self._sup_window,
        )
