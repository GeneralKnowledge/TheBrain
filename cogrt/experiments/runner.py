"""Compare two configurations from the same initial memory snapshot."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from ..backends.mock import MockBackend
from ..config import from_dict, load
from ..memory import MemoryStore
from ..runtime import Runtime
from . import metrics


def run_experiment_file(path: str | Path, out_dir: str = "results/experiments") -> dict[str, Any]:
    spec = tomllib.loads(Path(path).read_text())
    base_cfg = load(Path(path).parent / spec.get("base_config", "../../configs/mock.toml"))
    snap = spec.get("snapshot")
    runs = spec.get("runs", [])
    out = Path(out_dir) / Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for run in runs:
        name = run["name"]
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        if snap:
            shutil.copy2(snap, d / "memory.sqlite")
        else:
            MemoryStore(d / "memory.sqlite").close()
        overrides = run.get("config", {})
        cfg = from_dict(overrides, load(base_cfg.source_path or Path("configs/mock.toml")))
        cfg.runtime.data_dir = str(d)
        cfg.scheduler.interval_s = run.get("interval_s", 0.01)
        rt = Runtime(cfg, backend=MockBackend(seed=run.get("seed", 1)))
        rt.start(control_thread=False)
        if run.get("seed_memory"):
            for line in Path(run["seed_memory"]).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    rt.store.add("semantic", "note", line, "seed")
        rt.run(max_cycles=int(run.get("cycles", 3)))
        export = rt.store.export()
        cycles = rt.store.cycles(20)
        growth = metrics.memory_growth(rt.store)
        rt.shutdown()
        (d / "export.json").write_text(json.dumps(export, indent=1))
        results.append({"name": name, "growth": growth, "cycles": len(cycles)})
        MemoryStore(d / "memory.sqlite").close()
    comparison = {"runs": results}
    if len(results) == 2:
        comparison["delta_unresolved"] = (
            results[1]["growth"].get("unresolved", 0) - results[0]["growth"].get("unresolved", 0))
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    return {"comparison": comparison, "path": str(out)}
