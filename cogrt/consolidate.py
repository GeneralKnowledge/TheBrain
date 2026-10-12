"""Consolidation: durable-memory maintenance that does not need the model.

* merge near-duplicate open items (the newer one is superseded, the older
  one gains ``repeat_count``; confidence is unchanged)
* promote a hypothesis to a semantic conclusion only when a user-supplied or
  observed item supports it. Support from other generated items never
  promotes anything.
* apply retention limits
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .memory import MemoryStore
from .memory.text import content_words, jaccard

EXTERNAL = {"user", "observed", "seed"}


def merge_duplicates(store: MemoryStore, threshold: float, op: int | None) -> int:
    items = store.list("unresolved", "open", limit=500, order="id")
    merged = 0
    keep: list[tuple[Any, set[str]]] = []
    for it in items:
        if it.kind == "contradiction":
            continue
        words = content_words(it.content)
        dup = next((k for k, w in keep if k.kind == it.kind and jaccard(words, w) >= threshold), None)
        if dup is None:
            keep.append((it, words))
            continue
        store.revise(it.id, f"near duplicate of #{dup.id}", op, status="superseded")
        store.link(it.id, dup.id, "duplicate_of", operation_id=op)
        store._x("UPDATE items SET repeat_count=repeat_count+1 WHERE id=?", (dup.id,))
        merged += 1
    return merged


def promote_supported(store: MemoryStore, cfg: Config, op: int | None) -> list[int]:
    promoted = []
    for h in store.list("unresolved", "open", kind="hypothesis", limit=500):
        supporters = []
        for link in store.links(h.id):
            if link["relation"] != "supports" or link["dst"] != h.id:
                continue
            src = store.get(link["src"])
            if src and src.origin in EXTERNAL and src.status == "active":
                supporters.append(src)
        if not supporters:
            continue
        conf = min(0.8, (h.confidence or cfg.cycle.initial_hypothesis_confidence) + 0.2 * len(supporters))
        cid, created = store.add("semantic", "conclusion", h.content, "inferred", confidence=round(conf, 3),
                                 operation_id=op, source_ref=f"hypothesis:{h.id}",
                                 note="promoted: supported by " +
                                      ", ".join(f"#{s.id} ({s.origin})" for s in supporters))
        store.link(cid, h.id, "derived_from", operation_id=op)
        for s in supporters:
            store.link(s.id, cid, "supports", operation_id=op)
        store.revise(h.id, f"promoted to conclusion #{cid}", op, status="resolved")
        store.event("memory", "promoted", {"hypothesis": h.id, "conclusion": cid,
                                           "supporters": [s.id for s in supporters]})
        if created:
            promoted.append(cid)
    return promoted


def consolidate(store: MemoryStore, cfg: Config) -> dict[str, Any]:
    op = store.start_op("consolidate")
    with store.transaction():
        merged = merge_duplicates(store, cfg.cycle.duplicate_threshold, op)
        promoted = promote_supported(store, cfg, op)
    m = cfg.memory
    removed = store.apply_retention(m.max_episodic, m.episodic_retention_days, m.max_semantic, m.max_events)
    open_n = store.count("unresolved", "open")
    trimmed = 0
    if open_n > m.max_unresolved_open:
        for it in store.list("unresolved", "open", limit=open_n - m.max_unresolved_open, order="id"):
            if it.origin == "inferred":
                store.revise(it.id, "open-item limit reached", op, status="deferred")
                trimmed += 1
    result = {"merged": merged, "promoted": promoted, "removed": removed, "deferred_overflow": trimmed}
    changes = merged + len(promoted) + trimmed + sum(removed.values())
    store.end_op(op, "completed", result, state_changes=changes)
    if changes:
        store.event("memory", "consolidated", result)
    return result
