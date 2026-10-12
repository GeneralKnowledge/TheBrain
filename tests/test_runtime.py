from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from cogrt.backends.mock import MockBackend
from cogrt.runtime import AlreadyRunning, Runtime
from cogrt.seed_data import seed_store


def test_autonomous_cycles_without_user(cfg, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    cfg.scheduler.interval_s = 0.01
    cfg.scheduler.min_interval_s = 0.01
    cfg.scheduler.max_interval_s = 0.05
    cfg.scheduler.poll_s = 0.01
    rt = Runtime(cfg, backend=MockBackend(seed=2))
    rt.start(control_thread=False)
    seed_store(rt.store)
    rt.run(max_cycles=3)
    n = rt.store.count("unresolved") + rt.store.count("semantic")
    cn = rt.cycle_no
    rt.shutdown()
    assert n > 4
    assert cn == 3


def test_pause_persists(cfg, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    rt = Runtime(cfg, backend=MockBackend(seed=1))
    rt.start(control_thread=False)
    rt.store.set_meta("paused", True)
    rt.run(max_cycles=1, duration_s=0.2)
    assert rt.cycle_no == 0
    rt.shutdown()
    rt2 = Runtime(cfg, backend=MockBackend(seed=1))
    assert rt2.paused


def test_interrupt_during_cycle(cfg, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    cfg.scheduler.interval_s = 0.01
    backend = MockBackend(seed=1, latency_per_token_s=0.05)
    rt = Runtime(cfg, backend=backend)
    rt.start(control_thread=True)
    seed_store(rt.store)

    def interrupt_later() -> None:
        time.sleep(0.05)
        rt.store.enqueue("interrupt")

    threading.Thread(target=interrupt_later, daemon=True).start()
    rt.run(max_cycles=1, duration_s=5)
    cancelled = any(c["outcome"] == "cancelled" for c in rt.store.cycles(5))
    aborted = any(o["status"] == "aborted" for o in rt.store.operations(limit=5))
    rt.shutdown()
    assert cancelled or aborted


def test_single_instance_lock(cfg, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    rt = Runtime(cfg, backend=MockBackend(seed=1))
    rt.start(control_thread=False)
    rt2 = Runtime(cfg, backend=MockBackend(seed=1))
    with pytest.raises(AlreadyRunning):
        rt2.start(control_thread=False)
    rt.shutdown()


def test_conversation_does_not_replace_focus(cfg, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    rt = Runtime(cfg, backend=MockBackend(seed=1))
    rt.start(control_thread=False)
    seed_store(rt.store)
    rt.ws.focus.item_id = 999
    rt.ws.focus.policy = "memory"
    rt.converse("What about the river?", want_reply=False)
    assert rt.ws.focus.item_id == 999
    rt.shutdown()
