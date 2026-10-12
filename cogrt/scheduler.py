"""Runtime states, budgets and the decision whether another cycle is justified.

These states describe what the runtime is doing. They are not claims about
mental states.
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import SchedulerConfig
from .cycle.engine import CycleOutcome
from .memory import MemoryStore


class State(str, enum.Enum):
    DORMANT = "dormant"
    SCHEDULED = "scheduled"
    ASSOCIATING = "associating"
    REFLECTING = "reflecting"
    CONSOLIDATING = "consolidating"
    AVAILABLE = "available"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    STOPPED = "stopped"


POLICY_STATE = {
    "association": State.ASSOCIATING, "memory": State.ASSOCIATING, "random": State.ASSOCIATING,
    "unresolved": State.REFLECTING, "contradiction": State.REFLECTING,
}


class HourlyBudget:
    """Sliding one-hour window of cycles and generated tokens."""

    def __init__(self, cfg: SchedulerConfig, clock: Any = time.time) -> None:
        self.cfg = cfg
        self.clock = clock
        self.window: deque[tuple[float, int]] = deque()

    def load(self, store: MemoryStore) -> None:
        since = self.clock() - 3600
        for r in store._x("SELECT started_at, tokens_out FROM cycles WHERE started_at > ? AND model_calls > 0"
                          " ORDER BY started_at", (since,)):
            self.window.append((r["started_at"], r["tokens_out"]))

    def _trim(self) -> None:
        cutoff = self.clock() - 3600
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    def record(self, tokens: int) -> None:
        self.window.append((self.clock(), tokens))

    def usage(self) -> dict[str, int]:
        self._trim()
        return {"cycles": len(self.window), "tokens": sum(t for _, t in self.window)}

    def exhausted(self) -> str | None:
        u = self.usage()
        if u["cycles"] >= self.cfg.max_cycles_per_hour:
            return f"hourly cycle budget reached ({u['cycles']}/{self.cfg.max_cycles_per_hour})"
        if u["tokens"] >= self.cfg.token_budget_per_hour:
            return f"hourly token budget reached ({u['tokens']}/{self.cfg.token_budget_per_hour})"
        return None

    def seconds_until_free(self) -> float:
        self._trim()
        if not self.window:
            return 0.0
        return max(0.0, self.window[0][0] + 3600 - self.clock())


@dataclass
class Continuation:
    run_again: bool
    delay_s: float
    dormant: bool
    reason: str


class ContinuationPolicy:
    """Decide whether and when another cycle is justified.

    Productive cycles keep the base interval. Unproductive cycles back off
    exponentially. After ``dormant_after_unproductive`` unproductive cycles in
    a row, or when no policy has a target, the runtime becomes dormant until
    new input arrives or ``max_interval_s`` passes.
    """

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.interval = cfg.interval_s
        self.unproductive_streak = 0

    def reset(self) -> None:
        self.interval = self.cfg.interval_s
        self.unproductive_streak = 0

    def after(self, oc: CycleOutcome) -> Continuation:
        c = self.cfg
        if oc.outcome == "nothing_to_do":
            self.unproductive_streak += 1
            return Continuation(False, c.max_interval_s, True, "no applicable target: dormant")
        if oc.outcome == "cancelled":
            return Continuation(True, c.min_interval_s, False, f"cancelled ({oc.reason})")
        if oc.outcome == "failed":
            self.interval = min(c.max_interval_s, max(self.interval, c.min_interval_s) * c.backoff_factor)
            return Continuation(True, self.interval, False, f"failed: back off to {self.interval:.0f}s")
        if oc.productive:
            self.reset()
            return Continuation(True, max(c.min_interval_s, c.interval_s), False,
                                f"{oc.state_changes} change(s): keep base interval")
        self.unproductive_streak += 1
        if self.unproductive_streak >= c.dormant_after_unproductive:
            return Continuation(False, c.max_interval_s, True,
                                f"{self.unproductive_streak} unproductive cycles in a row: dormant")
        self.interval = min(c.max_interval_s, max(self.interval, c.min_interval_s) * c.backoff_factor)
        return Continuation(True, self.interval, False,
                            f"unproductive ({self.unproductive_streak}): back off to {self.interval:.0f}s")
