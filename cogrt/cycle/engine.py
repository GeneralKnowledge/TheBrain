"""One bounded internal cycle.

Stages (each replaceable through ``CycleEngine.stages``):

    select    choose a policy and a focus item, or decide there is nothing to do
    retrieve  fetch related memory (FTS + explicit links), not the last output
    generate  ask the model for up to N short candidates within the budget
    evaluate  deterministic checks (evaluate.assess)
    decide    retain / discard / defer, plus rule-based contradiction handling
    commit    write durable changes and provenance in one transaction

A cycle that produces no durable change is recorded as unproductive. The
model is never asked to address the user, and nothing is written to the
user-visible message channel.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..backends.base import Cancelled, CancelToken, GenerationRequest, ModelBackend
from ..config import Config
from ..hardware import process_rss_mb
from ..memory import ORIGIN_RANK, Item, MemoryStore
from ..workspace import Workspace
from . import evaluate, policies


@dataclass
class CycleState:
    cycle_id: int = 0
    cycle_no: int = 0
    selection: policies.Selection | None = None
    considered: dict[str, bool] = field(default_factory=dict)
    context: list[Item] = field(default_factory=list)
    prompt: str = ""
    example: str = ""
    raw: list[str] = field(default_factory=list)
    assessments: list[evaluate.Assessment] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    gen_latency_s: float = 0.0
    created: list[int] = field(default_factory=list)
    stage: str = ""
    finish: str = ""


@dataclass
class CycleOutcome:
    cycle_id: int
    policy: str | None
    focus_id: int | None
    outcome: str  # productive | unproductive | nothing_to_do | cancelled | failed
    state_changes: int
    retained: int
    revised: int
    discarded: int
    deferred: int
    model_calls: int
    tokens_in: int
    tokens_out: int
    duration_s: float
    reason: str = ""
    created: list[int] = field(default_factory=list)

    @property
    def productive(self) -> bool:
        return self.state_changes > 0


Stage = Callable[["CycleEngine", CycleState, CancelToken], None]


class CycleEngine:
    def __init__(self, store: MemoryStore, backend: ModelBackend, cfg: Config,
                 workspace: Workspace, rng: random.Random | None = None) -> None:
        self.store = store
        self.backend = backend
        self.cfg = cfg
        self.ws = workspace
        self.rng = rng or random.Random(cfg.model.seed)
        self.stages: list[tuple[str, Stage]] = [
            ("select", stage_select), ("retrieve", stage_retrieve), ("generate", stage_generate),
            ("evaluate", stage_evaluate), ("decide", stage_decide),
        ]
        self.forced_policy: str | None = None
        self.policy_weights: dict[str, float] | None = None
        # (policy, focus id) of a cycle that was preempted; selected again next time.
        self.resume: tuple[str, int] | None = None

    def _save_ws(self) -> None:
        self.store.set_meta("workspace", self.ws.to_dict())

    def run(self, cancel: CancelToken, cycle_no: int, sim_time: float | None = None) -> CycleOutcome:
        t0 = time.perf_counter()
        st = CycleState(cycle_no=cycle_no)
        st.cycle_id = self.store.start_cycle("pending", sim_time)
        token = cancel.child()
        timer = threading.Timer(self.cfg.scheduler.max_cycle_s,
                                lambda: token.cancel("cycle time budget exceeded"))
        timer.daemon = True
        timer.start()
        self.store.event("internal", "cycle_start", {"cycle": st.cycle_id, "n": cycle_no})
        try:
            for name, fn in self.stages:
                token.check()
                st.stage = name
                self.ws.focus.stage = name
                fn(self, st, token)
                self._save_ws()
                if st.selection is None and name == "select":
                    return self._finish(st, t0, "nothing_to_do", "no policy had an applicable target")
            token.check()  # last safe boundary before durable writes
            st.stage = "commit"
            changes = commit(self, st)
            self._save_ws()
            return self._finish(st, t0, "productive" if changes else "unproductive",
                                f"{changes} durable change(s)", changes)
        except Cancelled as exc:
            return self._finish(st, t0, "cancelled", str(exc) or token.reason)
        except Exception as exc:  # noqa: BLE001 - recorded and surfaced by the scheduler
            self.store.event("internal", "cycle_error", {"cycle": st.cycle_id, "error": repr(exc)})
            return self._finish(st, t0, "failed", repr(exc))
        finally:
            timer.cancel()

    def _finish(self, st: CycleState, t0: float, outcome: str, reason: str, changes: int = 0) -> CycleOutcome:
        sel = st.selection
        counts = {k: sum(1 for a in st.assessments if a.action == k)
                  for k in ("retain", "discard", "defer")}
        oc = CycleOutcome(
            cycle_id=st.cycle_id, policy=sel.policy if sel else None,
            focus_id=sel.focus.id if sel else None, outcome=outcome, state_changes=changes,
            retained=counts["retain"] if outcome in ("productive", "unproductive") else 0,
            revised=len(st.revisions), discarded=counts["discard"], deferred=counts["defer"],
            model_calls=st.model_calls, tokens_in=st.tokens_in, tokens_out=st.tokens_out,
            duration_s=time.perf_counter() - t0, reason=reason, created=st.created,
        )
        self.store.end_cycle(
            st.cycle_id, status="cancelled" if outcome == "cancelled" else "completed",
            outcome=outcome, policy=oc.policy, focus_item=oc.focus_id,
            productive=int(oc.productive), n_candidates=len(st.raw), n_retained=oc.retained,
            n_revised=oc.revised, n_discarded=oc.discarded, n_deferred=oc.deferred,
            model_calls=oc.model_calls, tokens_in=oc.tokens_in, tokens_out=oc.tokens_out,
            rss_mb=process_rss_mb(),
            detail={"reason": reason, "stage": st.stage, "considered": st.considered,
                    "selection_reason": sel.reason if sel else None,
                    "assessments": [a.to_dict() for a in st.assessments],
                    "revisions": st.revisions, "gen_latency_s": round(st.gen_latency_s, 3),
                    "backend": self.backend.name, "model_id": self.backend.model_id},
        )
        self.store.event("internal", "cycle_end", {
            "cycle": st.cycle_id, "outcome": outcome, "policy": oc.policy, "focus": oc.focus_id,
            "changes": changes, "calls": oc.model_calls, "tokens": oc.tokens_out,
            "duration_s": round(oc.duration_s, 3), "reason": reason})
        self.ws.focus.stage = None
        self._save_ws()
        return oc


# ---------------------------------------------------------------- stages
def stage_select(eng: CycleEngine, st: CycleState, _t: CancelToken) -> None:
    weights = eng.policy_weights or eng.cfg.cycle.policies
    resumed = None
    if eng.resume:
        policy, item_id = eng.resume
        eng.resume = None
        item = eng.store.get(item_id)
        # Two-item policies re-select their pair normally.
        if item and item.status in ("active", "open") and policy in ("unresolved", "memory", "random"):
            resumed = policies.Selection(policy, item, reason="resumed after interruption")
            resumed.output_kind = policies.template_for(resumed).output_kind
    if resumed:
        st.selection, st.considered = resumed, {"resumed": True}
    else:
        choice = policies.choose(eng.store, weights, eng.rng, st.cycle_no,
                                 eng.cfg.cycle.revisit_cooldown_cycles, eng.forced_policy,
                                 eng.cfg.cycle.max_inference_depth)
        st.selection, st.considered = choice.selection, choice.considered
    if st.selection:
        s = st.selection
        eng.ws.focus = type(eng.ws.focus)(item_id=s.focus.id, policy=s.policy, stage="select",
                                          cycle_id=st.cycle_id, content=s.focus.content[:200],
                                          secondary_id=s.secondary.id if s.secondary else None)
        eng.ws.push("recent_topics", {"id": s.focus.id, "cycle": st.cycle_no, "policy": s.policy})


def stage_retrieve(eng: CycleEngine, st: CycleState, _t: CancelToken) -> None:
    sel = st.selection
    assert sel is not None
    exclude = {sel.focus.id} | ({sel.secondary.id} if sel.secondary else set())
    if sel.contradiction:
        exclude.add(sel.contradiction.id)
    k = eng.cfg.cycle.context_items
    linked = [eng.store.get(i) for i in eng.store.linked_ids(sel.focus.id)]
    linked = [i for i in linked if i and i.id not in exclude and i.category != "episodic"][: k // 2]
    found = eng.store.search(sel.focus.content, limit=k, categories=("semantic", "unresolved"),
                             exclude=exclude | {i.id for i in linked}, touch=True)
    st.context = (linked + [i for i in found if i.kind != "contradiction"])[:k]
    eng.ws.set("relevant", [{"id": i.id, "content": i.content[:160], "origin": i.origin}
                            for i in st.context])
    st.prompt, st.example = policies.render(sel, st.context, k)


def _seed(base: int, cycle_no: int, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{cycle_no}:{k}".encode()).digest()[:4], "big")


def stage_generate(eng: CycleEngine, st: CycleState, token: CancelToken) -> None:
    g, s = eng.cfg.generation, eng.cfg.scheduler
    n = min(eng.cfg.cycle.n_candidates, s.max_model_calls_per_cycle)
    max_tokens = min(g.max_tokens, s.max_tokens_per_operation)
    for k in range(n):
        token.check()
        op = eng.store.start_op("generate", st.cycle_id, {"policy": st.selection.policy if st.selection else None,
                                                           "k": k, "prompt_chars": len(st.prompt)})
        req = GenerationRequest(prompt=st.prompt, max_tokens=max_tokens, temperature=g.temperature,
                                top_p=g.top_p, top_k=g.top_k, repeat_penalty=g.repeat_penalty,
                                stop=list(g.stop), seed=_seed(eng.cfg.model.seed, st.cycle_no, k),
                                cancel=token, tag="cycle")
        try:
            res = eng.backend.generate(req)
        except Exception as exc:  # noqa: BLE001
            eng.store.end_op(op, "failed", error=repr(exc))
            raise
        st.model_calls += 1
        st.tokens_in += res.prompt_tokens
        st.tokens_out += res.completion_tokens
        st.gen_latency_s += res.latency_s
        st.finish = res.finish_reason
        eng.store.end_op(op, "cancelled" if res.finish_reason == "cancelled" else "completed",
                         {"text": res.text, "finish": res.finish_reason, "latency_s": res.latency_s,
                          "rss_mb": res.rss_mb, "backend": res.backend},
                         res.prompt_tokens, res.completion_tokens)
        if res.finish_reason == "cancelled":
            token.check()
        st.raw.append(evaluate.clean(res.text, eng.cfg.cycle.max_chars))


def stage_evaluate(eng: CycleEngine, st: CycleState, _t: CancelToken) -> None:
    sel = st.selection
    assert sel is not None
    seen: set[str] = set()
    for text in st.raw:
        a = evaluate.assess(eng.store, eng.cfg.cycle, text, sel.focus, st.context, st.example,
                            sel.secondary)
        key = evaluate.norm_hash(text)
        if key in seen:
            a.duplicate_of = -1  # duplicate of another candidate in this cycle
        seen.add(key)
        st.assessments.append(a)


def stage_decide(eng: CycleEngine, st: CycleState, _t: CancelToken) -> None:
    for a in st.assessments:
        evaluate.decide(a, eng.cfg.cycle)
        if a.duplicate_of == -1:
            a.action, a.reason = "discard", "duplicate of another candidate in this cycle"
    eng.ws.set("proposals", [{"text": a.text[:160], "action": a.action, "reason": a.reason,
                              "novelty": round(a.novelty, 2), "relevance": round(a.relevance, 2)}
                             for a in st.assessments])
    for a in st.assessments:
        eng.ws.push("decisions", {"cycle": st.cycle_no, "action": a.action, "reason": a.reason,
                                  "text": a.text[:120]})


# ---------------------------------------------------------------- commit
def _contradiction_exists(store: MemoryStore, a: int, b: int) -> bool:
    for c in store.list("unresolved", kind="contradiction", limit=500):
        if {c.meta.get("a"), c.meta.get("b")} == {a, b}:
            return True
    return False


def open_contradiction(store: MemoryStore, a: Item, b: Item, op: int | None, note: str) -> int | None:
    if a.id == b.id or _contradiction_exists(store, a.id, b.id):
        return None
    cid, _ = store.add("unresolved", "contradiction", f"#{a.id} conflicts with #{b.id}: "
                       f"\"{a.content[:80]}\" vs \"{b.content[:80]}\"", "observed",
                       priority=0.8, meta={"a": a.id, "b": b.id}, operation_id=op,
                       note=note, dedupe=False)
    store.link(cid, a.id, "about", operation_id=op)
    store.link(cid, b.id, "about", operation_id=op)
    store.event("memory", "contradiction_opened", {"id": cid, "a": a.id, "b": b.id})
    return cid


def resolve_contradiction(eng: CycleEngine, st: CycleState, op: int) -> int:
    """Rule-based: higher-ranked provenance wins; equal-ranked claims stay open."""
    sel = st.selection
    assert sel and sel.contradiction and sel.secondary
    store, c = eng.store, sel.contradiction
    a, b = store.get(sel.focus.id), store.get(sel.secondary.id)
    if a is None or b is None:
        store.revise(c.id, "a party was deleted", op, status="resolved")
        return 1
    ra, rb = ORIGIN_RANK[a.origin], ORIGIN_RANK[b.origin]
    changes = 0
    if ra != rb:
        win, lose = (a, b) if ra > rb else (b, a)
        reason = f"contradicted by higher-ranked {win.origin} item #{win.id}"
        new_conf = round((lose.confidence or 0.5) * 0.5, 3)
        store.revise(lose.id, reason, op, status="superseded", confidence=new_conf)
        store.link(win.id, lose.id, "supersedes", operation_id=op)
        store.revise(c.id, f"resolved in favour of #{win.id}", op, status="resolved",
                     meta=c.meta | {"winner": win.id, "loser": lose.id})
        st.revisions.append({"item": lose.id, "status": "superseded", "reason": reason})
        changes += 2
    elif a.origin == "inferred":
        for it in (a, b):
            if it.confidence:
                store.revise(it.id, f"conflicts with #{(b if it is a else a).id}", op,
                             confidence=round(it.confidence * 0.7, 3))
                st.revisions.append({"item": it.id, "confidence": "lowered"})
                changes += 1
        status = "deferred" if c.times_explored >= 1 else "open"
        store.revise(c.id, "equal-rank inferred claims; confidence lowered", op,
                     status=status, priority=round(c.priority * 0.5, 3))
        changes += 1
    else:
        store.revise(c.id, "equal-rank observed/user claims: needs external review", op,
                     status="deferred", meta=c.meta | {"needs_review": True})
        st.revisions.append({"item": c.id, "status": "deferred", "reason": "needs review"})
        changes += 1
    store.mark_explored(c.id, st.cycle_no)
    return changes


_KIND = {"question": "question", "hypothesis": "hypothesis", "critique": "hypothesis",
         "resolution": "hypothesis", "association": "hypothesis"}


def commit(eng: CycleEngine, st: CycleState) -> int:
    sel = st.selection
    assert sel is not None
    store, cc = eng.store, eng.cfg.cycle
    op = store.start_op("commit", st.cycle_id, {"policy": sel.policy, "focus": sel.focus.id})
    changes = 0
    new_depth = 1 + max(policies.depth(sel.focus), policies.depth(sel.secondary) if sel.secondary else 0)
    with store.transaction():
        store.mark_explored(sel.focus.id, st.cycle_no)
        if sel.policy == "contradiction":
            changes += resolve_contradiction(eng, st, op)
        for a in st.assessments:
            if a.action != "retain":
                continue
            kind = _KIND[sel.output_kind]
            iid, created = store.add(
                "unresolved", kind, a.text, "inferred",
                confidence=cc.initial_hypothesis_confidence if kind == "hypothesis" else None,
                priority=0.5, operation_id=op, backend=eng.backend.name,
                model_id=eng.backend.model_id, source_ref=f"cycle:{st.cycle_id}",
                meta={"policy": sel.policy, "output_kind": sel.output_kind, "cycle": st.cycle_id,
                      "novelty": round(a.novelty, 3), "depth": new_depth},
                note=f"generated by {sel.policy} policy")
            if not created:
                a.action, a.reason = "discard", f"merged into existing #{iid}"
                continue
            st.created.append(iid)
            changes += 1
            store.link(iid, sel.focus.id, "derived_from", operation_id=op)
            if sel.secondary:
                store.link(iid, sel.secondary.id, "derived_from", operation_id=op)
            if sel.output_kind == "critique":
                store.link(iid, sel.focus.id, "contradicts" if sel.focus.id in a.conflicts_with
                           else "supports", operation_id=op)
            new = store.get(iid)
            assert new is not None
            for other_id in a.conflicts_with:
                other = store.get(other_id)
                if other and open_contradiction(store, new, other, op, "detected at commit"):
                    changes += 1
            store.event("memory", "retained", {"id": iid, "kind": kind, "text": a.text,
                                               "policy": sel.policy, "cycle": st.cycle_id})
        for a in st.assessments:
            if a.action == "defer":
                store.event("internal", "deferred", {"text": a.text, "reason": a.reason})
        focus = store.get(sel.focus.id)
        if (changes == 0 and focus and focus.category == "unresolved" and focus.status == "open"
                and focus.kind in ("question", "hypothesis") and focus.times_explored >= 3):
            store.revise(focus.id, f"explored {focus.times_explored}x without a retained result", op,
                         status="deferred", priority=round(focus.priority * 0.5, 3))
            # Recorded as a revision, but deciding to stop is not counted as productive work.
            st.revisions.append({"item": focus.id, "status": "deferred"})
    store.end_op(op, "completed", {"created": st.created, "revisions": st.revisions},
                 state_changes=changes)
    eng.ws.refresh(store)
    return changes
