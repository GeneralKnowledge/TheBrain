from __future__ import annotations

from pathlib import Path

import pytest

from cogrt import config as config_mod
from cogrt.backends.mock import MockBackend
from cogrt.memory import MemoryStore


@pytest.fixture
def cfg(tmp_path: Path) -> config_mod.Config:
    c = config_mod.load(overrides={
        "runtime": {"data_dir": str(tmp_path / "data")},
        "scheduler": {"interval_s": 0.01, "min_interval_s": 0.01, "max_interval_s": 0.05,
                      "max_cycle_s": 5.0, "poll_s": 0.01, "max_cycles_per_hour": 1000,
                      "token_budget_per_hour": 100000},
    })
    return c


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "m.sqlite")
    yield s
    s.close()


@pytest.fixture
def seeded(store: MemoryStore) -> MemoryStore:
    from cogrt.seed_data import seed_store
    seed_store(store)
    return store


@pytest.fixture
def mock() -> MockBackend:
    return MockBackend(seed=1)
