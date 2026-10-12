import time

import pytest

from cogrt.memory import MemoryStore


def test_add_dedupes_without_raising_confidence(store: MemoryStore) -> None:
    a, created = store.add("unresolved", "hypothesis", "Cold slows yeast.", "inferred", confidence=0.3)
    b, created2 = store.add("unresolved", "hypothesis", "cold   slows YEAST", "inferred", confidence=0.3)
    assert created and not created2 and a == b
    it = store.get(a)
    assert it.repeat_count == 1
    assert it.confidence == 0.3  # repetition is not evidence
    assert len(store.provenance(a)) == 2


def test_episodic_not_deduped(store: MemoryStore) -> None:
    a, _ = store.add("episodic", "interaction", "hello", "user")
    b, _ = store.add("episodic", "interaction", "hello", "user")
    assert a != b


def test_validation(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.add("bogus", "x", "y", "user")
    with pytest.raises(ValueError):
        store.add("semantic", "x", "y", "made-up")
    with pytest.raises(ValueError):
        store.add("semantic", "x", "   ", "user")


def test_search_relevance(seeded: MemoryStore) -> None:
    res = seeded.search("why is the tower clock slow", limit=3)
    assert res and "clock" in res[0].content.lower()
    assert seeded.search("") == []


def test_revise_records_history(store: MemoryStore) -> None:
    iid, _ = store.add("semantic", "note", "The bridge is old.", "seed", confidence=0.7)
    store.revise(iid, "new evidence", content="The bridge was rebuilt in 1990.", confidence=0.4)
    revs = store.revisions(iid)
    assert {r["field"] for r in revs} == {"content", "confidence"}
    assert store.search("rebuilt")[0].id == iid  # FTS follows the update
    assert store.search("old bridge", statuses=("active",)) == [] or \
        all(i.id != iid or "rebuilt" in i.content for i in store.search("old bridge"))


def test_forget_hard_deletes(store: MemoryStore) -> None:
    iid, _ = store.add("semantic", "note", "Secret thing to delete.", "user")
    assert store.forget(iid)
    assert store.get(iid) is None
    assert store.search("secret") == []
    assert store.provenance(iid) == []


def test_recover_marks_running_as_aborted(store: MemoryStore) -> None:
    op = store.start_op("generate")
    cyc = store.start_cycle("unresolved")
    rec = store.recover()
    assert rec["operations"] == 1 and rec["cycles"] == 1
    assert store.operations()[0]["status"] == "aborted"
    assert store.cycles()[0]["status"] == "aborted"
    assert op and cyc


def test_retention(tmp_path) -> None:
    now = [1_000_000.0]
    s = MemoryStore(tmp_path / "r.sqlite", clock=lambda: now[0])
    old, _ = s.add("episodic", "event", "old event", "observed")
    kept, _ = s.add("episodic", "interaction", "referenced", "user")
    h, _ = s.add("unresolved", "hypothesis", "derived thing", "inferred")
    s.link(h, kept, "derived_from")
    now[0] += 40 * 86400
    for i in range(5):
        s.add("episodic", "event", f"new event {i}", "observed")
    removed = s.apply_retention(max_episodic=3, episodic_retention_days=30, max_semantic=10, max_events=100)
    assert s.get(old) is None
    assert s.get(kept) is not None
    assert removed["episodic_old"] == 1
    assert s.count("episodic") <= 4


def test_persistence_across_reopen(tmp_path) -> None:
    p = tmp_path / "p.sqlite"
    s = MemoryStore(p)
    iid, _ = s.add("semantic", "note", "Persistent note about tides.", "seed")
    s.set_meta("paused", True)
    s.close()
    s2 = MemoryStore(p)
    assert s2.get(iid).content.startswith("Persistent")
    assert s2.get_meta("paused") is True


def test_commands_queue(store: MemoryStore) -> None:
    cid = store.enqueue("pause")
    assert store.pending_commands()[0]["id"] == cid
    store.finish_command(cid, "done", "ok")
    assert store.pending_commands() == []
    assert store.command(cid)["result"] == "ok"


def test_event_channels(store: MemoryStore) -> None:
    store.event("internal", "x")
    store.event("message", "reply", {"text": "hi"})
    assert [e["type"] for e in store.events("message")] == ["reply"]
    with pytest.raises(ValueError):
        store.event("nonsense", "x")


def test_snapshot(store: MemoryStore, tmp_path) -> None:
    store.add("semantic", "note", "Snapshot me.", "seed")
    store.snapshot(tmp_path / "snap.sqlite")
    s2 = MemoryStore(tmp_path / "snap.sqlite")
    assert s2.count("semantic") == 1
    _ = time
