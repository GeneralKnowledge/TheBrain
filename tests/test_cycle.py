from __future__ import annotations

import random

import pytest

from cogrt.backends.base import CancelToken
from cogrt.backends.mock import MockBackend
from cogrt.config import Config
from cogrt.cycle.engine import CycleEngine
from cogrt.cycle.evaluate import assess, decide, conflicts
from cogrt.memory import MemoryStore
from cogrt.workspace import Workspace

@pytest.fixture
def engine(seeded: MemoryStore, cfg: Config) -> CycleEngine:
    backend = MockBackend(seed=1)
    backend.load()
    ws = Workspace(max_items=8)
    return CycleEngine(seeded, backend, cfg, ws, random.Random(1))


def test_cycle_updates_memory_without_user_message(engine: CycleEngine, seeded: MemoryStore) -> None:
    before = seeded.count("unresolved", "open")
    oc = engine.run(CancelToken(), cycle_no=1)
    assert oc.outcome in ("productive", "unproductive", "nothing_to_do")
    msgs = [e for e in seeded.events("message")]
    assert msgs == []
    if oc.productive:
        assert seeded.count("unresolved", "open") >= before


def test_unproductive_when_no_targets(store: MemoryStore, cfg: Config) -> None:
    backend = MockBackend(seed=1)
    backend.load()
    eng = CycleEngine(store, backend, cfg, Workspace(), random.Random(0))
    oc = eng.run(CancelToken(), cycle_no=1)
    assert oc.outcome == "nothing_to_do"


def test_cancel_at_safe_boundary(engine: CycleEngine) -> None:
    tok = CancelToken()
    tok.cancel("test")
    oc = engine.run(tok, cycle_no=1)
    assert oc.outcome == "cancelled"


def test_evaluator_discards_duplicates(seeded: MemoryStore, cfg: Config) -> None:
    focus = seeded.list("unresolved", "open", kind="question")[0]
    text = focus.content
    a = assess(seeded, cfg.cycle, text, focus, [], "example unrelated text here")
    a = decide(a, cfg.cycle)
    assert a.action == "discard"


def test_conflicts_detects_negation() -> None:
    assert conflicts("The bridge was built in 1920.", "The bridge was not built in 1920.")
    assert not conflicts("Cats are mammals.", "Dogs are mammals.")


def test_contradiction_resolution(seeded: MemoryStore, cfg: Config) -> None:
    a, _ = seeded.add("semantic", "claim", "The tower is four minutes slow.", "user", confidence=0.8)
    b, _ = seeded.add("semantic", "claim", "The tower is not four minutes slow.", "inferred", confidence=0.3)
    from cogrt.cycle.engine import open_contradiction, resolve_contradiction, CycleState
    cid = open_contradiction(seeded, seeded.get(a), seeded.get(b), None, "test")
    assert cid
    backend = MockBackend(seed=1)
    eng = CycleEngine(seeded, backend, cfg, Workspace(), random.Random(0))
    sel = __import__("cogrt.cycle.policies", fromlist=["Selection"]).Selection(
        "contradiction", seeded.get(a), seeded.get(b), contradiction=seeded.get(cid))
    st = CycleState(cycle_id=1, cycle_no=1, selection=sel)
    changes = resolve_contradiction(eng, st, op=1)
    assert changes >= 1
    assert seeded.get(b).status == "superseded"
