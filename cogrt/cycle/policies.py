"""Selection policies and their prompt templates.

A policy answers two questions: *what* should the next cycle work on, and
*how* should the base model be asked about it. Base models continue text
rather than follow instructions, so templates are written as notebook
pages that end where the continuation should begin. Each template contains
one worked example; candidates that copy the example are discarded by the
evaluator.

A selector returns ``None`` when it has nothing worth doing. The runtime
treats "no policy applicable" as a reason to become dormant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..memory import Item, MemoryStore
from ..memory.text import content_words, jaccard


@dataclass
class Selection:
    policy: str
    focus: Item
    secondary: Item | None = None
    reason: str = ""
    output_kind: str = "hypothesis"  # hypothesis | question | critique | resolution | association
    contradiction: Item | None = None


@dataclass
class Template:
    header: str
    example: str
    body: str
    output_kind: str

    def render(self, **kw: str) -> str:
        return f"{self.header}\n\n{self.example}\n\n{self.body.format(**kw)}"


TEMPLATES: dict[str, Template] = {
    "unresolved": Template(
        header="Research notebook. Each open question is followed by one tentative idea that "
               "might help answer it. Ideas are tentative and are not facts.",
        example="Question: Why do some rivers flood after small storms?\nRelated notes:\n"
                "- Upstream soil was compacted by heavy machinery.\n"
                "Tentative idea: Compacted soil absorbs less water, so modest rain runs off quickly.",
        body="Question: {focus}\nRelated notes:\n{context}\nTentative idea:",
        output_kind="hypothesis",
    ),
    "reflect": Template(
        header="Research notebook. Each tentative hypothesis is followed by one critical check: a "
               "single sentence that either supports or undermines it.",
        example="Hypothesis: Plants near streetlights flower later in autumn.\nRelated notes:\n"
                "- Day length controls flowering in many species.\n"
                "Critical check: This would not hold for species whose flowering depends on temperature rather than light.",
        body="Hypothesis: {focus}\nRelated notes:\n{context}\nCritical check:",
        output_kind="critique",
    ),
    "contradiction": Template(
        header="Research notebook. When two notes conflict, the entry proposes one way both could be "
               "explained or which one needs checking.",
        example="Note A: The bridge was built in 1920.\nNote B: The bridge was not built until 1935.\n"
                "Possible resolution: The 1920 date may refer to an earlier wooden bridge on the same site.",
        body="Note A: {focus}\nNote B: {secondary}\nPossible resolution:",
        output_kind="resolution",
    ),
    "memory": Template(
        header="Research notebook. Each note is followed by one open question it raises.",
        example="Note: The library's oldest map shows a second harbour.\nRelated notes:\n"
                "- No harbour appears on modern maps.\n"
                "Open question: What happened to the second harbour?",
        body="Note: {focus}\nRelated notes:\n{context}\nOpen question:",
        output_kind="question",
    ),
    "association": Template(
        header="Research notebook. Each entry connects two unrelated notes with one specific remark.",
        example="Note A: Glaciers retreat faster after low-snowfall winters.\n"
                "Note B: Bees navigate using polarised light.\n"
                "Connection: Both depend on how sunlight interacts with a surface, so both could be studied with sky-light measurements.",
        body="Note A: {focus}\nNote B: {secondary}\nConnection:",
        output_kind="association",
    ),
    "random": Template(
        header="Research notebook. Each note is followed by one tentative idea it suggests.",
        example="Note: Old bread goes stale faster in the fridge.\nTentative idea: Cold may speed up the "
                "recrystallisation of starch.",
        body="Note: {focus}\nTentative idea:",
        output_kind="hypothesis",
    ),
}


def template_for(sel: Selection) -> Template:
    if sel.policy == "unresolved" and sel.focus.kind == "hypothesis":
        return TEMPLATES["reflect"]
    return TEMPLATES[sel.policy]


def depth(it: Item) -> int:
    return int(it.meta.get("depth", 1 if it.origin == "inferred" else 0))


ORIGIN_WEIGHT = {"user": 2.0, "seed": 1.5, "observed": 1.5, "inferred": 1.0}


def _eligible(it: Item, cycle_no: int, cooldown: int, max_depth: int = 2) -> bool:
    if depth(it) >= max_depth:
        return False
    return it.last_explored_cycle is None or cycle_no - it.last_explored_cycle > cooldown


def _w(it: Item) -> float:
    return ORIGIN_WEIGHT.get(it.origin, 1.0) / (1 + depth(it))


def _weighted(items: list[Item], rng: random.Random, weight: Callable[[Item], float]) -> Item | None:
    if not items:
        return None
    ws = [max(1e-6, weight(i)) for i in items]
    return rng.choices(items, ws, k=1)[0]


def select_unresolved(store: MemoryStore, rng: random.Random, cycle_no: int, cooldown: int,
                    max_depth: int = 2) -> Selection | None:
    items = [i for i in store.list("unresolved", "open", limit=200)
             if i.kind in ("question", "hypothesis") and _eligible(i, cycle_no, cooldown, max_depth)]
    pick = _weighted(items, rng, lambda i: _w(i) * i.priority / (1 + i.times_explored))
    if not pick:
        return None
    return Selection("unresolved", pick, reason=f"open {pick.kind}, explored {pick.times_explored}x")


def select_contradiction(store: MemoryStore, rng: random.Random, cycle_no: int, cooldown: int,
                    max_depth: int = 2) -> Selection | None:
    for c in store.list("unresolved", "open", kind="contradiction", limit=50, order="priority DESC"):
        if not _eligible(c, cycle_no, cooldown, 99):
            continue
        a, b = store.get(c.meta.get("a", -1)), store.get(c.meta.get("b", -1))
        if a and b:
            return Selection("contradiction", a, b, reason=f"contradiction #{c.id}",
                             contradiction=c)
    return None


def select_memory(store: MemoryStore, rng: random.Random, cycle_no: int, cooldown: int,
                    max_depth: int = 2) -> Selection | None:
    items = [i for i in store.list("semantic", "active", limit=200) +
             store.list("episodic", "active", kind="interaction", limit=50)
             if _eligible(i, cycle_no, cooldown, max_depth)]
    pick = _weighted(items, rng, lambda i: _w(i) * (1.0 + (i.confidence or 0.3)) / (1 + i.times_explored + i.access_count))
    if not pick:
        return None
    return Selection("memory", pick, reason="under-explored memory")


def select_association(store: MemoryStore, rng: random.Random, cycle_no: int, cooldown: int,
                    max_depth: int = 2) -> Selection | None:
    pool = [i for i in store.list("semantic", "active", limit=100) +
            store.list("unresolved", "open", kind="hypothesis", limit=50) if depth(i) < max_depth]
    if len(pool) < 2:
        return None
    for _ in range(12):
        a, b = rng.sample(pool, 2)
        if jaccard(content_words(a.content), content_words(b.content)) < 0.15 and \
                b.id not in store.linked_ids(a.id):
            return Selection("association", a, b, reason="previously unlinked pair")
    return None


def select_random(store: MemoryStore, rng: random.Random, cycle_no: int, cooldown: int,
                    max_depth: int = 2) -> Selection | None:
    pool = store.list(None, ("active", "open"), limit=300)
    pool = [i for i in pool if (i.category != "episodic" or i.kind == "interaction")
            and i.kind != "contradiction" and depth(i) < max_depth]
    if not pool:
        return None
    return Selection("random", rng.choice(pool), reason="random exploration")


SELECTORS: dict[str, Callable[..., Selection | None]] = {
    "unresolved": select_unresolved,
    "contradiction": select_contradiction,
    "memory": select_memory,
    "association": select_association,
    "random": select_random,
}


@dataclass
class PolicyChoice:
    selection: Selection | None
    considered: dict[str, bool] = field(default_factory=dict)


def choose(store: MemoryStore, weights: dict[str, float], rng: random.Random, cycle_no: int,
           cooldown: int, forced: str | None = None, max_depth: int = 2) -> PolicyChoice:
    """Weighted choice among the policies that currently have something to do."""
    names = [forced] if forced else [n for n, w in weights.items() if w > 0]
    options: list[tuple[str, Selection, float]] = []
    considered: dict[str, bool] = {}
    for n in names:
        if n not in SELECTORS:
            raise ValueError(f"unknown policy {n}")
        sel = SELECTORS[n](store, random.Random(rng.random()), cycle_no, cooldown, max_depth)
        considered[n] = sel is not None
        if sel:
            options.append((n, sel, weights.get(n, 1.0)))
    if not options:
        return PolicyChoice(None, considered)
    pick = rng.choices(options, [o[2] for o in options], k=1)[0]
    pick[1].output_kind = template_for(pick[1]).output_kind
    return PolicyChoice(pick[1], considered)


def context_block(items: list[Item], limit: int = 4) -> str:
    lines = [f"- {i.content[:200]}" for i in items[:limit]]
    return "\n".join(lines) if lines else "- (none)"


def render(sel: Selection, context: list[Item], limit: int) -> tuple[str, str]:
    """Return (prompt, example_text). The example is used for copy detection."""
    t = template_for(sel)
    kw: dict[str, Any] = {"focus": sel.focus.content[:300], "context": context_block(context, limit),
                          "secondary": sel.secondary.content[:300] if sel.secondary else ""}
    return t.render(**kw), t.example
