"""Default seed notes for demos and benchmarks."""

from __future__ import annotations

from pathlib import Path

from .memory import MemoryStore


def seed_store(store: MemoryStore, path: str | Path | None = None) -> int:
    if path:
        n = 0
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("?"):
                store.add("unresolved", "question", line, "seed", priority=0.7)
            else:
                store.add("semantic", "note", line, "seed", confidence=0.7)
            n += 1
        return n
    notes = [
        ("semantic", "note", "The clock tower runs four minutes slow every winter.", 0.7),
        ("semantic", "note", "Bread dough rises more slowly in the cold pantry.", 0.7),
        ("semantic", "note", "Copper roofs turn green over several decades.", 0.7),
        ("unresolved", "question", "Why does the clock tower lose time only in winter?", 0.7),
    ]
    for cat, kind, content, pri in notes:
        if cat == "semantic":
            store.add(cat, kind, content, "seed", confidence=pri)
        else:
            store.add(cat, kind, content, "seed", priority=pri)
    return len(notes)
