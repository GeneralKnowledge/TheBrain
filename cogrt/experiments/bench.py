"""Small benchmark suite (no paid APIs, no model self-assessment as sole metric)."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from ..backends.mock import MockBackend
from ..config import Config, load
from ..conversation import ingest
from ..cycle.engine import CycleEngine
from ..backends.base import CancelToken
from ..memory import MemoryStore
from ..runtime import Runtime
from ..workspace import Workspace
from . import metrics
from ..seed_data import seed_store


def _fresh_dir(base: Path, name: str) -> Path:
    d = base / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run_suite(cfg: Config | None = None, out_dir: str = "results/bench",
              only: list[str] | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg = cfg or load(root / "configs" / "mock.toml")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}

    def want(name: str) -> bool:
        return not only or name in only

    if want("scheduling"):
        d = _fresh_dir(out, "scheduling")
        c = load(root / "configs" / "mock.toml", overrides={"runtime": {"data_dir": str(d)}})
        c.scheduler.interval_s = 0.01
        rt = Runtime(c, backend=MockBackend(seed=3))
        rt.start(control_thread=False)
        seed_store(rt.store)
        rt.run(max_cycles=5)
        cycles = rt.store.cycles(10)
        episodic = rt.store.count("episodic", "active")
        rt.shutdown()
        recent = cycles[:5]
        cases["scheduling"] = {
            "pass": len(recent) >= 5 and all(c["model_calls"] > 0 for c in recent),
            "cycles": len(recent),
            "without_user_messages": episodic == 0,
        }

    if want("persistence"):
        d = _fresh_dir(out, "persistence")
        c = load(root / "configs" / "mock.toml", overrides={"runtime": {"data_dir": str(d)}})
        rt = Runtime(c, backend=MockBackend(seed=4))
        rt.start(control_thread=False)
        seed_store(rt.store)
        rt.run(max_cycles=2)
        n1 = rt.store.count("unresolved")
        rt.shutdown()
        rt2 = Runtime(c, backend=MockBackend(seed=4))
        rt2.start(control_thread=False)
        n2 = rt2.store.count("unresolved")
        rt2.shutdown()
        cases["persistence"] = {"pass": n2 >= n1, "before": n1, "after_restart": n2}

    if want("memory_recall"):
        store = MemoryStore(out / "recall.sqlite")
        seed_store(store)
        cases["memory_recall"] = {
            "pass": metrics.recall_hit(store, "clock tower slow winter", "clock"),
            "hits": [h.content for h in store.search("clock tower", limit=3)],
        }
        store.close()

    if want("contradiction_revision"):
        store = MemoryStore(out / "contradict.sqlite")
        ingest(store, "The bridge was built in 1920.")
        ingest(store, "Actually the bridge was not built until 1935.")
        open_c = store.list("unresolved", kind="contradiction")
        cases["contradiction_revision"] = {
            "pass": len(open_c) >= 1,
            "contradictions": len(open_c),
        }
        store.close()

    if want("repetition_novelty"):
        d = _fresh_dir(out, "novelty")
        store = MemoryStore(d / "m.sqlite")
        seed_store(store)
        backend = MockBackend(seed=99)
        backend.load()
        eng = CycleEngine(store, backend, load(root / "configs" / "mock.toml"), Workspace(), random.Random(99))
        texts = []
        for n in range(6):
            oc = eng.run(CancelToken(), cycle_no=n + 1)
            for cy in store.cycles(1):
                for a in cy["detail"].get("assessments", []):
                    if a.get("text"):
                        texts.append(a["text"])
        rep = metrics.repetition_rate(texts)
        nov = metrics.novel_vs_memory(store, texts)
        cases["repetition_novelty"] = {
            "pass": rep < 0.85 and nov > 0.1,
            "repetition_rate": round(rep, 3),
            "mean_novelty": round(nov, 3),
            "samples": len(texts),
        }
        store.close()

    if want("pause_interrupt"):
        d = _fresh_dir(out, "control")
        c = load(root / "configs" / "mock.toml", overrides={
            "runtime": {"data_dir": str(d)},
            "scheduler": {"interval_s": 0.01, "min_interval_s": 0.01, "max_interval_s": 0.05, "poll_s": 0.01},
        })
        rt = Runtime(c, backend=MockBackend(seed=5))
        rt.start(control_thread=False)
        seed_store(rt.store)
        rt.immediate("pause", {})
        cycles_before = len(rt.store.cycles(100))
        rt.run(max_cycles=5, duration_s=0.3)
        paused_ok = len(rt.store.cycles(100)) == cycles_before
        rt.immediate("resume", {})
        rt.run(max_cycles=1)
        resumed_ok = len(rt.store.cycles(100)) > cycles_before
        rt.shutdown()
        cases["pause_interrupt"] = {"pass": paused_ok and resumed_ok, "paused": paused_ok, "resumed": resumed_ok}

    summary = {
        "ts": time.time(),
        "backend": cfg.model.backend,
        "cases": cases,
        "failed": sum(1 for v in cases.values() if not v.get("pass")),
        "passed": sum(1 for v in cases.values() if v.get("pass")),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, "path": str(out)}
