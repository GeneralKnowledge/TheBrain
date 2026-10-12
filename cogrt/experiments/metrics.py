from __future__ import annotations

from typing import Any

from ..memory import MemoryStore
from ..memory.text import content_words, distinct_n, jaccard, norm_hash


def memory_growth(store: MemoryStore) -> dict[str, int]:
    return {cat: store.count(cat) for cat in ("semantic", "unresolved", "episodic")}


def repetition_rate(texts: list[str], n: int = 2) -> float:
    return 1.0 - distinct_n(texts, n) if texts else 0.0


def recall_hit(store: MemoryStore, query: str, expected_substring: str) -> bool:
    hits = store.search(query, limit=5)
    return any(expected_substring.lower() in h.content.lower() for h in hits)


def state_stability(before: dict[str, Any], after: dict[str, Any]) -> float:
    """Fraction of semantic norm_hashes unchanged between exports."""
    bh = {norm_hash(i["content"]) for i in before.get("items", []) if i["category"] == "semantic"}
    ah = {norm_hash(i["content"]) for i in after.get("items", []) if i["category"] == "semantic"}
    if not bh:
        return 1.0
    return len(bh & ah) / len(bh)


def novel_vs_memory(store: MemoryStore, texts: list[str]) -> float:
    if not texts:
        return 0.0
    scores = []
    for t in texts:
        words = content_words(t)
        sims = [s for _, s in store.near_duplicates(t, threshold=0.0, limit=5)]
        scores.append(1.0 - max(sims, default=0.0))
    return sum(scores) / len(scores)
