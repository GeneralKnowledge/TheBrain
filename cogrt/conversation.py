"""Optional conversation with the user.

A message influences internal state (episodic record, user claims and
questions with provenance ``user``, contradiction detection) but does not
replace the current focus: new items are added with a priority and the
selection policies decide when to look at them. A reply is generated only
when asked for, and it goes to the ``message`` channel, separate from
internal events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .backends.base import CancelToken, GenerationRequest, ModelBackend
from .config import Config
from .cycle.engine import open_contradiction
from .cycle.evaluate import conflicts, find_conflicts
from .memory import MemoryStore
from .memory.text import content_words, jaccard

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Ingested:
    episode_id: int
    claims: list[int] = field(default_factory=list)
    questions: list[int] = field(default_factory=list)
    contradictions: list[int] = field(default_factory=list)
    supports: list[tuple[int, int]] = field(default_factory=list)


def ingest(store: MemoryStore, text: str, op: int | None = None) -> Ingested:
    text = text.strip()
    ep, _ = store.add("episodic", "interaction", text, "user", operation_id=op,
                      meta={"role": "user"}, dedupe=False, priority=0.6)
    out = Ingested(ep)
    for sent in (s.strip() for s in _SENT.split(text) if s.strip()):
        if len(content_words(sent)) < 2:
            continue
        if sent.endswith("?"):
            qid, created = store.add("unresolved", "question", sent, "user", priority=0.9,
                                     operation_id=op, source_ref=f"episode:{ep}")
            if not created:
                store.revise(qid, "asked again by user", op, priority=0.9, status="open")
            store.link(qid, ep, "derived_from", operation_id=op)
            out.questions.append(qid)
            continue
        cid, _ = store.add("semantic", "claim", sent, "user", confidence=0.8, priority=0.7,
                           operation_id=op, source_ref=f"episode:{ep}",
                           note="supplied by the user; not independently verified")
        store.link(cid, ep, "derived_from", operation_id=op)
        out.claims.append(cid)
        claim = store.get(cid)
        assert claim is not None
        for other in find_conflicts(store, sent, exclude={cid}):
            c = open_contradiction(store, claim, other, op, "user claim conflicts with memory")
            if c:
                out.contradictions.append(c)
        words = content_words(sent)
        for h in store.search(sent, limit=8, categories=("unresolved",), statuses=("open",)):
            if h.kind == "hypothesis" and not conflicts(sent, h.content) and \
                    jaccard(words, content_words(h.content)) >= 0.4:
                store.link(cid, h.id, "supports", operation_id=op)
                out.supports.append((cid, h.id))
    store.event("memory", "user_input", {"episode": ep, "claims": out.claims,
                                         "questions": out.questions,
                                         "contradictions": out.contradictions})
    return out


REPLY_TEMPLATE = (
    "Below is a log of messages exchanged with a research system. The system's own notes "
    "are listed first; notes marked (tentative) are its unverified hypotheses.\n\n"
    "Notes:\n{notes}\n\nUser: {message}\nSystem:"
)


def reply(store: MemoryStore, backend: ModelBackend, cfg: Config, text: str,
          cancel: CancelToken | None = None, op: int | None = None) -> dict[str, Any]:
    ctx = store.search(text, limit=cfg.cycle.context_items, categories=("semantic", "unresolved"),
                       touch=True)
    notes = "\n".join(f"- {i.content[:200]}" + (" (tentative)" if i.origin == "inferred" else "")
                      for i in ctx) or "- (none)"
    prompt = REPLY_TEMPLATE.format(notes=notes, message=text)
    g = cfg.generation
    res = backend.generate(GenerationRequest(
        prompt=prompt, max_tokens=min(g.max_tokens * 2, cfg.scheduler.max_tokens_per_operation * 2),
        temperature=g.temperature, top_p=g.top_p, top_k=g.top_k, repeat_penalty=g.repeat_penalty,
        stop=["\nUser:", "\n\n"], seed=cfg.model.seed, cancel=cancel, tag="reply"))
    answer = res.text.strip()
    if answer:
        rid, _ = store.add("episodic", "interaction", answer, "inferred", operation_id=op,
                           meta={"role": "system"}, dedupe=False, backend=backend.name,
                           model_id=backend.model_id)
    store.event("message", "reply", {"text": answer, "context": [i.id for i in ctx],
                                     "finish": res.finish_reason})
    return {"text": answer, "tokens_in": res.prompt_tokens, "tokens_out": res.completion_tokens,
            "latency_s": res.latency_s, "finish": res.finish_reason}
