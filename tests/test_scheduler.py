from cogrt.config import SchedulerConfig
from cogrt.cycle.engine import CycleOutcome
from cogrt.scheduler import ContinuationPolicy, HourlyBudget


def test_hourly_budget() -> None:
    cfg = SchedulerConfig(max_cycles_per_hour=2, token_budget_per_hour=100)
    b = HourlyBudget(cfg, clock=lambda: 1000.0)
    b.record(40)
    b.record(50)
    assert b.exhausted() is not None
    u = b.usage()
    assert u["cycles"] == 2 and u["tokens"] == 90


def test_continuation_backoff_and_dormant() -> None:
    cfg = SchedulerConfig(interval_s=10, min_interval_s=5, max_interval_s=100, backoff_factor=2,
                          dormant_after_unproductive=2)
    p = ContinuationPolicy(cfg)
    unprod = CycleOutcome(1, "x", 1, "unproductive", 0, 0, 0, 0, 0, 0, 0, 0, 0.1)
    d1 = p.after(unprod)
    assert d1.run_again and not d1.dormant
    d2 = p.after(unprod)
    assert d2.dormant and not d2.run_again
    prod = CycleOutcome(2, "x", 1, "productive", 2, 1, 0, 0, 0, 1, 10, 10, 0, 0.1)
    d3 = p.after(prod)
    assert d3.run_again and p.unproductive_streak == 0
