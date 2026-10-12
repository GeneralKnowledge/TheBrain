"""``cogrt``: terminal interface suitable for SSH.

The runtime (``cogrt run``) is the primary process. Every other subcommand is
a client that reads the SQLite database directly or queues a command for
the runtime. Clients work when the runtime is not running: inspection is
read-only, and pause/resume change the persistent flag.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import config as config_mod
from .hardware import detect, recommend
from .memory import MemoryStore


def _cfg(args: argparse.Namespace) -> config_mod.Config:
    path = args.config or os.environ.get("COGRT_CONFIG")
    overrides = {"runtime": {"data_dir": args.data}} if getattr(args, "data", None) else None
    return config_mod.load(path, overrides)


def _store(cfg: config_mod.Config) -> MemoryStore:
    return MemoryStore(cfg.data_path / "memory.sqlite")


def daemon_running(cfg: config_mod.Config) -> bool:
    lock = cfg.data_path / "runtime.lock"
    if not lock.exists():
        return False
    with open(lock, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def _print(obj: Any, as_json: bool = False) -> None:
    if as_json or not isinstance(obj, (list, dict)):
        print(json.dumps(obj, indent=2, default=str) if not isinstance(obj, str) else obj)
        return
    print(json.dumps(obj, indent=2, default=str))


def _wait(store: MemoryStore, cmd_id: int, timeout: float) -> dict[str, Any] | None:
    end = time.time() + timeout
    while time.time() < end:
        c = store.command(cmd_id)
        if c and c["status"] in ("done", "failed"):
            return c
        time.sleep(0.25)
    return store.command(cmd_id)


def _send(cfg: config_mod.Config, cmd: str, args: dict[str, Any] | None = None,
          wait: float = 10.0) -> dict[str, Any] | None:
    store = _store(cfg)
    cid = store.enqueue(cmd, args or {})
    if not daemon_running(cfg):
        print(f"(runtime not running; command #{cid} queued for the next start)")
        return None
    return _wait(store, cid, wait) if wait else {"id": cid, "status": "pending"}


def fmt_item(it: Any) -> str:
    conf = "" if it.confidence is None else f" c={it.confidence:.2f}"
    return f"#{it.id:<5} {it.category[:4]}/{it.kind:<13} {it.status:<10} {it.origin:<8}{conf}  {it.content[:110]}"


def fmt_event(e: dict[str, Any]) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
    p = e["payload"]
    if e["type"] == "cycle_end":
        body = (f"cycle {p.get('cycle')} {p.get('outcome')} policy={p.get('policy')} focus=#{p.get('focus')} "
                f"changes={p.get('changes')} calls={p.get('calls')} tok={p.get('tokens')} "
                f"{p.get('duration_s')}s :: {p.get('reason')}")
    elif e["type"] == "retained":
        body = f"retained #{p.get('id')} [{p.get('kind')}] {p.get('text')}"
    elif e["type"] == "reply":
        body = f"SYSTEM: {p.get('text')}"
    elif e["type"] == "state":
        body = f"{p.get('from')} -> {p.get('to')} {p.get('why', '')}"
    else:
        body = json.dumps(p, default=str)[:200]
    return f"{ts} [{e['channel']:<10}] {e['type']:<20} {body}"


# ------------------------------------------------------------------ commands
def cmd_hw(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    rep = detect(cfg.data_path)
    out = rep.to_dict() | {"recommendation": recommend(rep)}
    _print(out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import logging

    from .runtime import AlreadyRunning, Runtime

    cfg = _cfg(args)
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(cfg.data_path / "runtime.log"), level=cfg.runtime.log_level,
                        format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    rt = Runtime(cfg)
    try:
        rec = rt.start()
    except AlreadyRunning as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rt.install_signal_handlers()
    print(f"runtime pid={os.getpid()} data={cfg.data_path} backend={rt.backend.name} "
          f"model={rt.backend.model_id} paused={rt.paused} recovered={rec}", flush=True)
    try:
        rt.run(max_cycles=args.max_cycles, duration_s=args.duration, idle_exit=args.idle_exit)
    finally:
        rt.shutdown()
    print("runtime stopped", flush=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    store = _store(cfg)
    st = store.get_meta("status", {}) or {}
    running = daemon_running(cfg)
    st = st | {"running": running, "paused_flag": bool(store.get_meta("paused", False)),
               "memory": store.stats()}
    if not running:
        st["state"] = "stopped"
    if args.json:
        _print(st, True)
        return 0
    print(f"state: {st.get('state')}  running: {running}  paused: {st['paused_flag']}  "
          f"cycle: {st.get('cycle_no')}  next in: {st.get('next_due_in_s')}s")
    if st.get("dormant_reason"):
        print(f"dormant because: {st['dormant_reason']}")
    if st.get("current"):
        print(f"current operation: {st['current']}")
    f = st.get("focus") or {}
    if f.get("item_id"):
        print(f"focus: #{f['item_id']} via {f.get('policy')} stage={f.get('stage')} :: {f.get('content', '')[:90]}")
    if st.get("last_outcome"):
        lo = st["last_outcome"]
        print(f"last cycle: {lo['outcome']} ({lo['changes']} changes) — {lo['decision']}")
    b = st.get("backend") or {}
    print(f"backend: {b.get('backend')} {b.get('model_id')} loaded={st.get('backend_loaded')}  "
          f"rss={st.get('rss_mb')}MB peak={st.get('peak_rss_mb')}MB")
    if st.get("budget_hour"):
        lim = st.get("limits", {})
        print(f"budget (last hour): {st['budget_hour']['cycles']}/{lim.get('max_cycles_per_hour')} cycles, "
              f"{st['budget_hour']['tokens']}/{lim.get('token_budget_per_hour')} tokens")
    m = st["memory"]
    print(f"memory: {json.dumps(m['items'])}  links={m['links']} revisions={m['revisions']} "
          f"cycles={m['cycles']} db={m['db_bytes'] // 1024}KB")
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")
            cmd_status(args)
            print()
            store = _store(_cfg(args))
            for e in store.events(limit=args.lines):
                print(fmt_event(e)[:200])
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_events(args: argparse.Namespace) -> int:
    store = _store(_cfg(args))
    last = 0
    evs = store.events(channel=args.channel, limit=args.limit)
    for e in evs:
        print(json.dumps(e, default=str) if args.json else fmt_event(e))
        last = e["id"]
    while args.follow:
        time.sleep(1)
        for e in store.events(channel=args.channel, after_id=last, limit=200):
            print(json.dumps(e, default=str) if args.json else fmt_event(e), flush=True)
            last = e["id"]
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    store = _store(_cfg(args))
    _print(store.get_meta("workspace", {}) or {})
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    store = _store(cfg)
    if args.action == "list":
        for it in store.list(args.category, args.status, args.kind, limit=args.limit):
            print(fmt_item(it))
    elif args.action == "search":
        for it in store.search(" ".join(args.terms), limit=args.limit,
                               statuses=("active", "open", "deferred", "resolved", "superseded")):
            print(fmt_item(it))
    elif args.action == "show":
        it = store.get(int(args.terms[0]))
        if not it:
            print("not found")
            return 1
        _print({"item": it.__dict__, "provenance": store.provenance(it.id),
                "links": store.links(it.id), "revisions": store.revisions(it.id)})
    elif args.action == "forget":
        if not args.yes:
            print("refusing to delete without --yes")
            return 1
        for t in args.terms:
            print(f"#{t}: {'deleted' if store.forget(int(t), 'cli forget') else 'not found'}")
    elif args.action == "add":
        cat, kind = args.category or "semantic", args.kind or ("question" if args.category == "unresolved" else "note")
        iid, created = store.add(cat, kind, " ".join(args.terms), args.origin,
                                 confidence=0.8 if args.origin != "inferred" else 0.3,
                                 note="added via cli")
        print(f"#{iid} {'created' if created else 'merged with existing'}")
    elif args.action == "stats":
        _print(store.stats())
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Load seed notes: one per line; lines ending with '?' become open questions."""
    store = _store(_cfg(args))
    n = 0
    for line in Path(args.file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("?"):
            store.add("unresolved", "question", line, "seed", priority=0.7, note=f"seed file {args.file}")
        else:
            store.add("semantic", "note", line, "seed", confidence=0.7, note=f"seed file {args.file}")
        n += 1
    print(f"seeded {n} items")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    text = " ".join(args.text)
    if not daemon_running(cfg):
        from .conversation import ingest
        res = ingest(_store(cfg), text)
        print(f"(runtime not running: message stored without a reply) {res.__dict__}")
        return 0
    r = _send(cfg, "say", {"text": text, "reply": not args.no_reply}, wait=args.wait)
    if r is None:
        return 1
    if r["status"] == "done":
        res = r["result"]
        if res.get("reply") is not None:
            print(f"SYSTEM: {res['reply']}")
        print(f"(state updates: claims={res['claims']} questions={res['questions']} "
              f"contradictions={res['contradictions']})")
    else:
        print(f"command #{r['id']} status={r['status']} result={r.get('result')}")
    return 0


def cmd_simple(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    name = args.cmd_name
    if name in ("pause", "resume") and not daemon_running(cfg):
        store = _store(cfg)
        store.set_meta("paused", name == "pause")
        store.event("control", name, {"offline": True})
        print(f"{name}d (persistent flag set; runtime not running)")
        return 0
    if name == "stop" and args.now:
        (cfg.data_path / "STOP").touch()
        lock = cfg.data_path / "runtime.lock"
        pid = lock.read_text().strip() if lock.exists() else ""
        if pid.isdigit() and daemon_running(cfg):
            os.kill(int(pid), signal.SIGTERM)
            print(f"STOP file written and SIGTERM sent to {pid}")
        else:
            print("STOP file written")
        return 0
    payload: dict[str, Any] = {}
    if name == "set":
        for kv in args.pairs:
            k, _, v = kv.partition("=")
            payload[k] = v
    if name == "cycle" and args.policy:
        payload["policy"] = args.policy
    r = _send(cfg, name, payload, wait=args.wait if hasattr(args, "wait") else 10)
    if r:
        print(f"{r['status']}: {r.get('result')}")
    return 0


def cmd_cycles(args: argparse.Namespace) -> int:
    store = _store(_cfg(args))
    for c in reversed(store.cycles(args.limit)):
        if args.json:
            print(json.dumps(c, default=str))
            continue
        d = c["detail"]
        print(f"cycle {c['id']:<5} {c['outcome'] or c['status']:<14} policy={c['policy']:<13} "
              f"focus=#{c['focus_item']} cand={c['n_candidates']} kept={c['n_retained']} "
              f"disc={c['n_discarded']} def={c['n_deferred']} calls={c['model_calls']} tok={c['tokens_out']}"
              f" :: {d.get('reason')}")
        if args.verbose:
            for a in d.get("assessments", []):
                print(f"      {a['action']:<8} {a['reason']:<45} {a['text'][:90]!r}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    store = _store(cfg)
    base = (cfg.data_path / cfg.security.export_dir)
    base.mkdir(parents=True, exist_ok=True)
    dest = base / Path(args.name or f"export-{int(time.time())}.json").name
    dest.write_text(json.dumps(store.export(), indent=1, default=str))
    print(dest)
    return 0


def cmd_console(args: argparse.Namespace) -> int:
    """Line-oriented console. Plain text is sent as a message; /commands control the runtime."""
    help_text = ("/status /events [n] /ws /mem <terms> /show <id> /open /cycles /pause /resume "
                 "/interrupt /cycle [policy] /set k=v /stop /quit — anything else is sent as a message")
    print(help_text)
    while True:
        try:
            line = input("cogrt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if not line.startswith("/"):
            main(_base_args(args) + ["say", line])
            continue
        parts = shlex.split(line[1:])
        c, rest = parts[0], parts[1:]
        mapping = {"status": ["status"], "events": ["events", "--limit", rest[0] if rest else "20"],
                   "ws": ["workspace"], "mem": ["memory", "search", *rest], "show": ["memory", "show", *rest],
                   "open": ["memory", "list", "--category", "unresolved", "--status", "open"],
                   "cycles": ["cycles", "-v"], "pause": ["pause"], "resume": ["resume"],
                   "interrupt": ["interrupt"], "stop": ["stop"], "set": ["set", *rest],
                   "cycle": ["cycle", *(["--policy", rest[0]] if rest else [])]}
        if c in ("quit", "exit", "q"):
            return 0
        if c not in mapping:
            print(help_text)
            continue
        main(_base_args(args) + mapping[c])


def _base_args(args: argparse.Namespace) -> list[str]:
    out = []
    if args.config:
        out += ["-c", args.config]
    if getattr(args, "data", None):
        out += ["--data", args.data]
    return out


def cmd_bench(args: argparse.Namespace) -> int:
    from .experiments.bench import run_suite
    report = run_suite(_cfg(args) if args.config else None, out_dir=args.out, only=args.only)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    from .experiments.runner import run_experiment_file
    report = run_experiment_file(args.file, out_dir=args.out)
    print(json.dumps(report["comparison"], indent=2, default=str))
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    from .sim.engine import run_simulation_file
    report = run_simulation_file(args.file, out_dir=args.out)
    print(json.dumps(report["summary"], indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cogrt", description=__doc__.split("\n")[0])
    p.add_argument("-c", "--config", help="TOML config (or $COGRT_CONFIG)")
    p.add_argument("--data", help="override [runtime] data_dir")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hw", help="detect hardware and suggest a configuration").set_defaults(fn=cmd_hw)
    r = sub.add_parser("run", help="start the runtime in the foreground")
    r.add_argument("--max-cycles", type=int)
    r.add_argument("--duration", type=float, help="seconds")
    r.add_argument("--idle-exit", action="store_true", help="exit when dormant with no pending input")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="runtime status report")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    t = sub.add_parser("top", help="refreshing status + recent events")
    t.add_argument("--interval", type=float, default=2.0)
    t.add_argument("--lines", type=int, default=15)
    t.set_defaults(fn=cmd_top, json=False)
    e = sub.add_parser("events", help="internal events, messages, memory updates, experiment logs")
    e.add_argument("--channel", choices=["internal", "message", "memory", "experiment", "control"])
    e.add_argument("--limit", type=int, default=30)
    e.add_argument("-f", "--follow", action="store_true")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_events)
    sub.add_parser("workspace", help="current working state").set_defaults(fn=cmd_workspace)
    m = sub.add_parser("memory", help="inspect and edit memory")
    m.add_argument("action", choices=["list", "search", "show", "forget", "add", "stats"])
    m.add_argument("terms", nargs="*")
    m.add_argument("--category", choices=["episodic", "semantic", "unresolved"])
    m.add_argument("--status")
    m.add_argument("--kind")
    m.add_argument("--origin", default="user", choices=["user", "seed", "observed", "inferred"])
    m.add_argument("--limit", type=int, default=30)
    m.add_argument("--yes", action="store_true", help="confirm deletion")
    m.set_defaults(fn=cmd_memory)
    sd = sub.add_parser("seed", help="load seed notes from a text file")
    sd.add_argument("file")
    sd.set_defaults(fn=cmd_seed)
    sy = sub.add_parser("say", help="send a message (influences state; optional reply)")
    sy.add_argument("text", nargs="+")
    sy.add_argument("--no-reply", action="store_true")
    sy.add_argument("--wait", type=float, default=120.0)
    sy.set_defaults(fn=cmd_say)
    for name, hlp in [("pause", "suspend autonomous processing (persistent)"), ("resume", "resume"),
                      ("interrupt", "cancel the current operation at the next safe point"),
                      ("stop", "stop the runtime"), ("consolidate", "run consolidation now"),
                      ("snapshot", "copy the database into the export directory")]:
        sp = sub.add_parser(name, help=hlp)
        sp.set_defaults(fn=cmd_simple, cmd_name=name, wait=30)
        if name == "stop":
            sp.add_argument("--now", action="store_true", help="emergency stop: STOP file + SIGTERM")
    st = sub.add_parser("set", help="adjust budgets, e.g. set interval_s=120 max_cycles_per_hour=10")
    st.add_argument("pairs", nargs="+")
    st.set_defaults(fn=cmd_simple, cmd_name="set", wait=10)
    cy = sub.add_parser("cycle", help="request one internal cycle now")
    cy.add_argument("--policy", choices=["unresolved", "contradiction", "memory", "association", "random"])
    cy.add_argument("--wait", type=float, default=300)
    cy.set_defaults(fn=cmd_simple, cmd_name="cycle")
    cl = sub.add_parser("cycles", help="process history")
    cl.add_argument("--limit", type=int, default=20)
    cl.add_argument("-v", "--verbose", action="store_true")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(fn=cmd_cycles)
    ex = sub.add_parser("export", help="export memory as JSON into the export directory")
    ex.add_argument("--name")
    ex.set_defaults(fn=cmd_export)
    sub.add_parser("console", help="interactive line console").set_defaults(fn=cmd_console)
    b = sub.add_parser("bench", help="benchmark suite")
    b.add_argument("--out", default="results/bench")
    b.add_argument("--only", nargs="*")
    b.set_defaults(fn=cmd_bench)
    x = sub.add_parser("experiment", help="run an experiment definition (TOML)")
    x.add_argument("file")
    x.add_argument("--out", default="results/experiments")
    x.set_defaults(fn=cmd_experiment)
    si = sub.add_parser("sim", help="run an accelerated population simulation (TOML)")
    si.add_argument("file")
    si.add_argument("--out", default="results/sim")
    si.set_defaults(fn=cmd_sim)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
