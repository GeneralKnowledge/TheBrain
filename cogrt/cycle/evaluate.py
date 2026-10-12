"""Deterministic examination of generated candidates.

The evaluator never asks the model whether its own output is good. It uses
lexical checks that are cheap, reproducible and inspectable:

* length bounds and in-text repetition (degenerate loops)
* exact and near duplication against durable memory
* copying of the prompt's worked example or of retrieved context
* novelty relative to memory and context
* relevance to the focus item
* conflict with existing items (negation polarity or differing numbers on
  heavily overlapping content)

These heuristics are deliberately simple baselines. Their limits are
documented in docs/STATUS.md, and the stage can be replaced.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import CycleConfig
from ..memory import Item, MemoryStore
from ..memory.text import content_words, internal_repetition, jaccard, negated, norm_hash, tokens

_NUM = re.compile(r"\d+(?:\.\d+)?")


def clean(text: str, max_chars: int) -> str:
    text = text.strip().split("\n", 1)[0].strip()
    text = text.strip("\"'` ")
    for marker in ("Question:", "Note:", "Note A:", "Note B:", "Hypothesis:", "Related notes:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text


@dataclass
class Assessment:
    text: str
    length_ok: bool
    repetition: float
    duplicate_of: int | None
    near_duplicate_of: int | None
    near_similarity: float
    copies_prompt: bool
    example_leakage: float
    novelty: float
    relevance: float
    conflicts_with: list[int] = field(default_factory=list)
    supports: bool = False
    action: str = "discard"  # retain | revise | discard | defer
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def conflicts(a: str, b: str, min_overlap: float = 0.5) -> bool:
    """True when two statements are about the same thing but disagree on polarity or numbers."""
    wa = {w for w in content_words(a) if not _NUM.fullmatch(w)}
    wb = {w for w in content_words(b) if not _NUM.fullmatch(w)}
    if jaccard(wa, wb) < min_overlap:
        return False
    if negated(a) != negated(b):
        return True
    na, nb = set(_NUM.findall(a)), set(_NUM.findall(b))
    return bool(na and nb and na != nb)


def find_conflicts(store: MemoryStore, text: str, exclude: set[int] | None = None,
                   limit: int = 10) -> list[Item]:
    exclude = exclude or set()
    out = []
    for it in store.search(text, limit=limit, categories=("semantic", "unresolved")):
        if it.id in exclude or it.kind == "contradiction":
            continue
        if conflicts(text, it.content):
            out.append(it)
    return out


def assess(store: MemoryStore, cfg: CycleConfig, text: str, focus: Item,
           context: list[Item], example: str, secondary: Item | None = None) -> Assessment:
    words = content_words(text)
    focus_words = content_words(focus.content) | (content_words(secondary.content) if secondary else set())
    length_ok = cfg.min_chars <= len(text) <= cfg.max_chars and len(tokens(text)) >= 3
    rep = internal_repetition(text)

    exact = store.find_exact("unresolved", text) or store.find_exact("semantic", text)
    near = store.near_duplicates(text, threshold=cfg.duplicate_threshold)
    near = [(i, s) for i, s in near if not exact or i.id != exact.id]

    refs = [focus.content] + ([secondary.content] if secondary else []) + [c.content for c in context]
    copies = norm_hash(text) == norm_hash(focus.content) or \
        jaccard(words, content_words(example)) >= 0.6 or \
        any(jaccard(words, content_words(r)) >= cfg.duplicate_threshold for r in refs)

    ref_words = set().union(*(content_words(r) for r in refs))
    leaked = (words & content_words(example)) - ref_words
    leakage = len(leaked) / max(1, len(words))

    sims = [jaccard(words, content_words(r)) for r in refs]
    sims += [s for _, s in store.near_duplicates(text, threshold=0.0, limit=10)]
    novelty = 1.0 - max(sims, default=0.0)
    relevance = len(words & focus_words) / max(1, len(words)) if words else 0.0

    a = Assessment(text=text, length_ok=length_ok, repetition=rep,
                   duplicate_of=exact.id if exact else None,
                   near_duplicate_of=near[0][0].id if near else None,
                   near_similarity=near[0][1] if near else 0.0,
                   copies_prompt=copies, example_leakage=leakage, novelty=novelty,
                   relevance=relevance)
    if a.length_ok and not copies:
        a.conflicts_with = [i.id for i in find_conflicts(store, text, exclude={focus.id})]
        if focus.kind == "hypothesis":
            if conflicts(text, focus.content, min_overlap=0.3):
                a.conflicts_with.append(focus.id)
            else:
                a.supports = relevance > 0
    return a


def decide(a: Assessment, cfg: CycleConfig) -> Assessment:
    """Retain, discard or defer. Generating text is never sufficient for retention."""
    if not a.text:
        a.action, a.reason = "discard", "empty output"
    elif not a.length_ok:
        a.action, a.reason = "discard", "outside length bounds"
    elif a.repetition > 0.3:
        a.action, a.reason = "discard", f"degenerate repetition {a.repetition:.2f}"
    elif a.copies_prompt:
        a.action, a.reason = "discard", "copies the prompt, example or context"
    elif a.example_leakage > cfg.max_example_leakage:
        a.action, a.reason = "discard", f"borrows from the template example ({a.example_leakage:.2f})"
    elif a.duplicate_of is not None:
        a.action, a.reason = "discard", f"exact duplicate of #{a.duplicate_of}"
    elif a.near_duplicate_of is not None:
        a.action, a.reason = "discard", f"near duplicate of #{a.near_duplicate_of} ({a.near_similarity:.2f})"
    elif a.relevance < cfg.relevance_threshold:
        a.action, a.reason = "discard", f"irrelevant to focus ({a.relevance:.2f})"
    elif a.novelty < cfg.novelty_threshold:
        a.action, a.reason = "defer", f"low novelty {a.novelty:.2f}; kept in workspace only"
    else:
        a.action, a.reason = "retain", f"novelty {a.novelty:.2f}, relevance {a.relevance:.2f}"
    return a
