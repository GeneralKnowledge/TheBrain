"""Lightweight population simulation.

Many agents advance logical time with mock or cached inference. Actual model
calls are counted separately from scheduled cycles.
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.mock import MockBackend
from ..config import load
from ..cycle.engine import CycleEngine
from ..backends.base import CancelToken
from ..memory import MemoryStore
from ..workspace import Workspace


@dataclass
class AgentSim:
    agent_id: int
    store: MemoryStore
    engine: CycleEngine
    sim_time: float = 0.0
    scheduled_cycles: int = 0
    model_invocations: int = 0
    tokens_out: int = 0
    state_changes: int = 0


@dataclass
class SimReport:
    agents: int
    sim_time: float
    wall_s: float
    scheduled_cycles: int
    model_invocations: int
    tokens_out: int
    state_changes: int
    per_agent: list[dict[str, Any]] = field(default_factory=list)


def run_simulation(spec: dict[str, Any], out_dir: Path) -> SimReport:
    import time

    root = Path(__file__).resolve().parents[2]
    cfg = load(root / "configs" / spec.get("config", "mock.toml"))
    n_agents = int(spec.get("agents", 10))
    sim_duration = float(spec.get("sim_time", 3600))
    tick = float(spec.get("tick_s", 60))
    invoke_rate = float(spec.get("model_invoke_rate", 0.05))  # fraction of ticks that call the model
    use_model = bool(spec.get("use_model", False))
    t0 = time.perf_counter()
    agents: list[AgentSim] = []
    for i in range(n_agents):
        db = out_dir / f"agent_{i}.sqlite"
        store = MemoryStore(db)
        for note in spec.get("seed_notes", spec.get("seed", []))[:3]:
            store.add("semantic", "note", note, "seed")
        backend = MockBackend(seed=i, latency_per_token_s=0) if use_model else MockBackend(seed=i)
        if not use_model:
            backend.script = ["Cached association for simulation."] * 1000
        backend.load()
        eng = CycleEngine(store, backend, cfg, Workspace(), random.Random(i))
        agents.append(AgentSim(i, store, eng))
    sim_t = 0.0
    rng = random.Random(spec.get("rng_seed", spec.get("seed", 0)))
    while sim_t < sim_duration:
        for ag in agents:
            ag.scheduled_cycles += 1
            ag.sim_time = sim_t
            if use_model and rng.random() < invoke_rate:
                oc = ag.engine.run(CancelToken(), cycle_no=ag.scheduled_cycles, sim_time=sim_t)
                ag.model_invocations += oc.model_calls
                ag.tokens_out += oc.tokens_out
                ag.state_changes += oc.state_changes
        sim_t += tick
    wall = time.perf_counter() - t0
    per = []
    for ag in agents:
        per.append({"id": ag.agent_id, "scheduled_cycles": ag.scheduled_cycles,
                    "model_invocations": ag.model_invocations, "tokens_out": ag.tokens_out,
                    "state_changes": ag.state_changes})
        ag.store.close()
    return SimReport(n_agents, sim_duration, wall, sum(a.scheduled_cycles for a in agents),
                     sum(a.model_invocations for a in agents), sum(a.tokens_out for a in agents),
                     sum(a.state_changes for a in agents), per)


def run_simulation_file(path: str | Path, out_dir: str = "results/sim") -> dict[str, Any]:
    spec = tomllib.loads(Path(path).read_text())
    out = Path(out_dir) / Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    rep = run_simulation(spec, out)
    summary = {
        "agents": rep.agents,
        "sim_time": rep.sim_time,
        "wall_s": round(rep.wall_s, 3),
        "scheduled_cycles": rep.scheduled_cycles,
        "model_invocations": rep.model_invocations,
        "tokens_out": rep.tokens_out,
        "state_changes": rep.state_changes,
        "ratio_model_to_cycles": round(rep.model_invocations / max(1, rep.scheduled_cycles), 4),
    }
    (out / "report.json").write_text(json.dumps({"summary": summary, "per_agent": rep.per_agent}, indent=2))
    return {"summary": summary, "path": str(out)}
