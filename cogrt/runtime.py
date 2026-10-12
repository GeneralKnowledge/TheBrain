"""The runtime process: the primary application.

The main loop decides when the model runs. A control thread reads commands
from the database so that SSH clients can pause, interrupt or stop the
process while a model call is in progress. Conversation is one input among
others and is handled between cycles.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import queue
import random
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .backends import create_backend
from .backends.base import CancelToken, ModelBackend
from .config import Config
from .consolidate import consolidate
from .conversation import ingest, reply
from .cycle.engine import CycleEngine, CycleOutcome
from .hardware import peak_rss_mb, process_rss_mb
from .memory import MemoryStore
from .scheduler import POLICY_STATE, ContinuationPolicy, HourlyBudget, State
from .workspace import Workspace

log = logging.getLogger("cogrt")

IMMEDIATE = {"pause", "resume", "interrupt", "stop", "set"}
DEFERRED = {"say", "cycle", "consolidate", "snapshot"}
SETTABLE = {
    "interval_s": float, "min_interval_s": float, "max_interval_s": float, "max_cycle_s": float,
    "max_tokens_per_operation": int, "max_model_calls_per_cycle": int, "max_cycles_per_hour": int,
    "token_budget_per_hour": int, "autonomous": bool,
}


class AlreadyRunning(RuntimeError):
    pass


class Runtime:
    def __init__(self, cfg: Config, backend: ModelBackend | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.data = cfg.data_path
        self.data.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.store = MemoryStore(self.data / "memory.sqlite", clock=clock)
        apply_overrides(cfg, self.store)
        self.backend = backend or create_backend(cfg)
        w = cfg.workspace
        self.ws = Workspace.from_dict(self.store.get_meta("workspace", {}) or {}, w.max_items_per_slot,
                                      w.recent_topics)
        self.cycle_no = int(self.store.get_meta("cycle_no", 0))
        self.rng = random.Random(f"{cfg.model.seed}:{self.cycle_no}")
        self.engine = CycleEngine(self.store, self.backend, cfg, self.ws, self.rng)
        self.cont = ContinuationPolicy(cfg.scheduler)
        self.budget = HourlyBudget(cfg.scheduler, clock)
        self.root = CancelToken()
        self.current: CancelToken | None = None
        self.current_kind = ""
        self.inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.state = State.PAUSED if self.paused else State.SCHEDULED
        self.next_due = clock() + cfg.scheduler.min_interval_s
        self.dormant_reason = ""
        self.last_outcome: dict[str, Any] | None = None
        self.started_at = clock()
        self._lock_fh: Any = None
        self._control: threading.Thread | None = None
        self.stop_file = self.data / "STOP"

    # -------------------------------------------------------------- state
    @property
    def paused(self) -> bool:
        return bool(self.store.get_meta("paused", False))

    def set_state(self, s: State, why: str = "") -> None:
        if s != self.state:
            self.store.event("control", "state", {"from": self.state.value, "to": s.value, "why": why})
            self.state = s
        self.publish_status()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value, "paused": self.paused, "pid": os.getpid(),
            "heartbeat": self.clock(), "started_at": self.started_at, "cycle_no": self.cycle_no,
            "next_due_in_s": round(max(0.0, self.next_due - self.clock()), 1),
            "dormant_reason": self.dormant_reason, "current": self.current_kind,
            "last_outcome": self.last_outcome, "budget_hour": self.budget.usage(),
            "limits": {k: getattr(self.cfg.scheduler, k) for k in SETTABLE},
            "unproductive_streak": self.cont.unproductive_streak,
            "backend": self.backend.info(), "backend_loaded": self.backend.loaded,
            "rss_mb": round(process_rss_mb(), 1), "peak_rss_mb": round(peak_rss_mb(), 1),
            "focus": self.ws.to_dict()["focus"],
        }

    def publish_status(self) -> None:
        self.store.set_meta("status", self.status())

    # ---------------------------------------------------------- lifecycle
    def acquire_lock(self) -> None:
        fh = open(self.data / "runtime.lock", "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            raise AlreadyRunning(f"another runtime holds {self.data / 'runtime.lock'}") from exc
        fh.write(str(os.getpid()))
        fh.flush()
        self._lock_fh = fh

    def start(self, control_thread: bool = True) -> dict[str, int]:
        self.acquire_lock()
        if self.stop_file.exists():
            self.stop_file.unlink()
        recovered = self.store.recover()
        self.budget.load(self.store)
        self.store.event("control", "runtime_start", {"pid": os.getpid(), "recovered": recovered,
                                                      "cycle_no": self.cycle_no, "paused": self.paused,
                                                      "backend": self.backend.info()})
        if any(recovered.values()):
            log.warning("recovered after unclean shutdown: %s", recovered)
        if control_thread:
            self._control = threading.Thread(target=self._control_loop, name="control", daemon=True)
            self._control.start()
        self.publish_status()
        return recovered

    def shutdown(self) -> None:
        self.root.cancel("shutdown")
        self.set_state(State.STOPPED, "shutdown")
        self.store.event("control", "runtime_stop", {"cycle_no": self.cycle_no})
        self.publish_status()
        if self._control:
            self._control.join(timeout=5)
        try:
            self.backend.unload()
        except Exception:  # noqa: BLE001
            log.exception("unload failed")
        if self._lock_fh:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
            self._lock_fh = None
        self.store.close()

    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            if self.root.cancelled:
                os._exit(130)  # second signal: leave immediately
            self.request_stop(f"signal {signum}")
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def request_stop(self, reason: str) -> None:
        self.root.cancel(reason)
        if self.current:
            self.current.cancel(reason)

    # ------------------------------------------------------------ control
    def _control_loop(self) -> None:
        while not self.root.cancelled:
            try:
                self.poll_control()
            except Exception:  # noqa: BLE001
                log.exception("control loop error")
            time.sleep(self.cfg.scheduler.poll_s)

    def poll_control(self) -> None:
        if self.stop_file.exists():
            self.request_stop("STOP file present")
            return
        for cmd in self.store.pending_commands():
            name = cmd["cmd"]
            if name in IMMEDIATE:
                try:
                    result = self.immediate(name, cmd["args"])
                    self.store.finish_command(cmd["id"], "done", result)
                except Exception as exc:  # noqa: BLE001
                    self.store.finish_command(cmd["id"], "failed", repr(exc))
            elif name in DEFERRED:
                self.store._x("UPDATE commands SET status='running' WHERE id=?", (cmd["id"],))
                self.inbox.put(cmd)
                if name == "say" and self.cfg.scheduler.user_preempts and self.current_kind == "cycle" \
                        and self.current:
                    self.preempt()
            else:
                self.store.finish_command(cmd["id"], "failed", f"unknown command {name}")
        self.store.set_meta("heartbeat", self.clock())

    def preempt(self) -> None:
        f = self.ws.focus
        if f.policy and f.item_id is not None:
            self.engine.resume = (f.policy, f.item_id)
        if self.current:
            self.current.cancel("preempted by user message")

    def immediate(self, name: str, args: dict[str, Any]) -> Any:
        if name == "pause":
            self.store.set_meta("paused", True)
            if self.current and self.current_kind == "cycle":
                self.current.cancel("paused")
            self.set_state(State.PAUSED, "pause command")
            return "paused"
        if name == "resume":
            self.store.set_meta("paused", False)
            self.cont.reset()
            self.next_due = self.clock() + self.cfg.scheduler.min_interval_s
            self.set_state(State.SCHEDULED, "resume command")
            return "resumed"
        if name == "interrupt":
            if self.current:
                self.current.cancel("interrupted by user")
                return f"interrupting {self.current_kind}"
            return "nothing running"
        if name == "stop":
            self.request_stop("stop command")
            return "stopping"
        if name == "set":
            changed = {}
            for k, v in args.items():
                if k not in SETTABLE:
                    raise ValueError(f"cannot set {k}; allowed: {sorted(SETTABLE)}")
                typ = SETTABLE[k]
                val = (str(v).lower() in ("1", "true", "yes", "on")) if typ is bool else typ(v)
                if typ in (int, float) and val <= 0:
                    raise ValueError(f"{k} must be positive")
                setattr(self.cfg.scheduler, k, val)
                changed[k] = val
            self.store.set_meta("overrides", {**(self.store.get_meta("overrides", {}) or {}), **changed})
            self.store.event("control", "budget_changed", changed)
            self.publish_status()
            return changed
        raise ValueError(name)

    # ------------------------------------------------------------ main loop
    def run(self, max_cycles: int | None = None, duration_s: float | None = None,
            idle_exit: bool = False) -> None:
        end = self.clock() + duration_s if duration_s else None
        ran = 0
        while not self.root.cancelled:
            if self.stop_file.exists():
                self.request_stop("STOP file present")
                break
            did = self.step()
            if did == "cycle":
                ran += 1
                if max_cycles is not None and ran >= max_cycles:
                    break
            if end and self.clock() >= end:
                break
            if idle_exit and self.state == State.DORMANT and self.inbox.empty():
                break
            if did is None:
                time.sleep(self.cfg.scheduler.poll_s)

    def step(self) -> str | None:
        """One scheduling decision. Returns what was done, or None when idle."""
        try:
            cmd = self.inbox.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd:
            self.run_deferred(cmd)
            return cmd["cmd"]
        if self.paused:
            self.set_state(State.PAUSED, "persistent pause flag")
            return None
        if not self.cfg.scheduler.autonomous:
            self.set_state(State.DORMANT, "autonomous processing disabled")
            return None
        now = self.clock()
        if now < self.next_due:
            self.set_state(State.DORMANT if self.dormant_reason else State.SCHEDULED)
            return None
        why = self.budget.exhausted()
        if why:
            self.next_due = now + max(self.cfg.scheduler.min_interval_s, self.budget.seconds_until_free())
            self.dormant_reason = why
            self.set_state(State.DORMANT, why)
            return None
        self.run_cycle()
        return "cycle"

    def run_cycle(self) -> CycleOutcome:
        self.dormant_reason = ""
        self.cycle_no += 1
        self.store.set_meta("cycle_no", self.cycle_no)
        tok = self.root.child()
        self.current, self.current_kind = tok, "cycle"
        self.set_state(State.SCHEDULED, f"cycle {self.cycle_no}")
        orig_select = self.engine.stages[0][1]

        def select_and_announce(eng: CycleEngine, st: Any, t: CancelToken) -> None:
            orig_select(eng, st, t)
            if st.selection:
                self.set_state(POLICY_STATE.get(st.selection.policy, State.REFLECTING),
                               f"{st.selection.policy}: {st.selection.reason}")

        self.engine.stages[0] = ("select", select_and_announce)
        try:
            oc = self.engine.run(tok, self.cycle_no)
        finally:
            self.engine.stages[0] = ("select", orig_select)
            self.current, self.current_kind = None, ""
        if oc.model_calls:
            self.budget.record(oc.tokens_out)
        if oc.outcome == "cancelled":
            self.set_state(State.INTERRUPTED, oc.reason)
        dec = self.cont.after(oc)
        self.next_due = self.clock() + dec.delay_s
        self.dormant_reason = dec.reason if dec.dormant else ""
        self.last_outcome = {"cycle": oc.cycle_id, "outcome": oc.outcome, "policy": oc.policy,
                             "changes": oc.state_changes, "reason": oc.reason,
                             "decision": dec.reason, "next_in_s": round(dec.delay_s, 1)}
        self.store.event("internal", "continuation", self.last_outcome)
        if dec.dormant or self.cycle_no % self.cfg.scheduler.consolidate_every == 0:
            self.run_consolidation()
        if self.paused:
            self.set_state(State.PAUSED, "paused")
        else:
            self.set_state(State.DORMANT if dec.dormant else State.SCHEDULED, dec.reason)
        return oc

    def run_consolidation(self) -> dict[str, Any]:
        prev = self.state
        self.set_state(State.CONSOLIDATING, "consolidation")
        res = consolidate(self.store, self.cfg)
        self.ws.refresh(self.store)
        self.store.set_meta("workspace", self.ws.to_dict())
        self.set_state(prev if prev != State.CONSOLIDATING else State.SCHEDULED)
        return res

    def run_deferred(self, cmd: dict[str, Any]) -> None:
        name, args = cmd["cmd"], cmd["args"]
        try:
            if name == "say":
                result = self.converse(args.get("text", ""), want_reply=args.get("reply", True))
            elif name == "cycle":
                if args.get("policy"):
                    self.engine.forced_policy = args["policy"]
                try:
                    oc = self.run_cycle()
                finally:
                    self.engine.forced_policy = None
                result = {"outcome": oc.outcome, "changes": oc.state_changes, "created": oc.created,
                          "reason": oc.reason}
            elif name == "consolidate":
                result = self.run_consolidation()
            elif name == "snapshot":
                dest = self._safe_path(args.get("name", f"snapshot-{int(self.clock())}.sqlite"))
                self.store.snapshot(dest)
                result = str(dest)
            else:
                raise ValueError(name)
            self.store.finish_command(cmd["id"], "done", result)
        except Exception as exc:  # noqa: BLE001
            log.exception("command %s failed", name)
            self.store.finish_command(cmd["id"], "failed", repr(exc))

    def _safe_path(self, name: str) -> Path:
        base = (self.data / self.cfg.security.export_dir).resolve()
        dest = (base / Path(name).name).resolve()
        if base not in dest.parents:
            raise ValueError("export path escapes the export directory")
        return dest

    def converse(self, text: str, want_reply: bool = True) -> dict[str, Any]:
        prev = self.state
        self.set_state(State.AVAILABLE, "user message")
        op = self.store.start_op("user_message", None, {"chars": len(text)})
        ing = ingest(self.store, text, op)
        out: dict[str, Any] = {"episode": ing.episode_id, "claims": ing.claims,
                               "questions": ing.questions, "contradictions": ing.contradictions}
        tokens_in = tokens_out = 0
        if want_reply:
            tok = self.root.child()
            self.current, self.current_kind = tok, "reply"
            try:
                r = reply(self.store, self.backend, self.cfg, text, tok, op)
            finally:
                self.current, self.current_kind = None, ""
            out["reply"] = r["text"]
            tokens_in, tokens_out = r["tokens_in"], r["tokens_out"]
        self.store.end_op(op, "completed", out, tokens_in, tokens_out,
                          len(ing.claims) + len(ing.questions) + len(ing.contradictions))
        self.ws.pending_user.append({"episode": ing.episode_id, "text": text[:200]})
        del self.ws.pending_user[:-self.cfg.workspace.max_items_per_slot]
        self.ws.refresh(self.store)
        self.store.set_meta("workspace", self.ws.to_dict())
        # New input is a reason to leave dormancy, but the current focus is preserved.
        self.cont.reset()
        self.dormant_reason = ""
        self.next_due = min(self.next_due, self.clock() + self.cfg.scheduler.min_interval_s)
        self.set_state(State.PAUSED if self.paused else (prev if prev not in (State.AVAILABLE, State.DORMANT)
                                                          else State.SCHEDULED))
        return out


def apply_overrides(cfg: Config, store: MemoryStore) -> None:
    """Re-apply budget changes made at runtime so they survive a restart."""
    for k, v in (store.get_meta("overrides", {}) or {}).items():
        if k in SETTABLE:
            setattr(cfg.scheduler, k, SETTABLE[k](v))


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
