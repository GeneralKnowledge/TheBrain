"""Durable memory, process history and control queue in a single SQLite file.

Categories
    episodic    records of interactions and significant internal events
    semantic    retained concepts, relations and conclusions
    unresolved  open questions, hypotheses and contradictions

Every item carries an ``origin`` (seed | user | observed | inferred) and a
provenance trail. Generated text is stored as ``inferred`` and never becomes
a verified fact because it was repeated: repetition increments
``repeat_count`` but never ``confidence``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .text import content_words, fts_query, jaccard, norm_hash

SCHEMA_VERSION = 1

CATEGORIES = {"episodic", "semantic", "unresolved"}
ORIGINS = {"seed", "user", "observed", "inferred"}
# Higher rank wins when two claims conflict.
ORIGIN_RANK = {"inferred": 0, "seed": 1, "observed": 2, "user": 2}
CHANNELS = {"internal", "message", "memory", "experiment", "control"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    norm_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    origin TEXT NOT NULL,
    confidence REAL,
    priority REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    repeat_count INTEGER NOT NULL DEFAULT 0,
    times_explored INTEGER NOT NULL DEFAULT 0,
    last_explored_cycle INTEGER,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS items_cat_status ON items(category, status);
CREATE INDEX IF NOT EXISTS items_hash ON items(category, norm_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(content, content='items', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE OF content ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, content) VALUES ('delete', old.id, old.content);
  INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY,
    src INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    operation_id INTEGER,
    created_at REAL NOT NULL,
    UNIQUE(src, dst, relation)
);
CREATE INDEX IF NOT EXISTS links_dst ON links(dst);
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    operation_id INTEGER,
    backend TEXT,
    model_id TEXT,
    note TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS prov_item ON provenance(item_id);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT NOT NULL,
    operation_id INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS rev_item ON revisions(item_id);
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    sim_time REAL,
    policy TEXT,
    focus_item INTEGER,
    status TEXT NOT NULL,
    outcome TEXT,
    productive INTEGER NOT NULL DEFAULT 0,
    n_candidates INTEGER NOT NULL DEFAULT 0,
    n_retained INTEGER NOT NULL DEFAULT 0,
    n_revised INTEGER NOT NULL DEFAULT 0,
    n_discarded INTEGER NOT NULL DEFAULT 0,
    n_deferred INTEGER NOT NULL DEFAULT 0,
    model_calls INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    rss_mb REAL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY,
    cycle_id INTEGER,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    duration_s REAL,
    input TEXT NOT NULL DEFAULT '{}',
    output TEXT NOT NULL DEFAULT '{}',
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    state_changes INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ops_cycle ON operations(cycle_id);
CREATE INDEX IF NOT EXISTS ops_status ON operations(status);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    channel TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_channel ON events(channel, id);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    cmd TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    done_at REAL
);
"""


@dataclass
class Item:
    id: int
    category: str
    kind: str
    content: str
    status: str
    origin: str
    confidence: float | None
    priority: float
    created_at: float
    updated_at: float
    access_count: int
    repeat_count: int
    times_explored: int
    last_explored_cycle: int | None
    meta: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def brief(self) -> dict[str, Any]:
        return {"id": self.id, "category": self.category, "kind": self.kind,
                "status": self.status, "origin": self.origin, "confidence": self.confidence,
                "content": self.content}


def _row_item(row: sqlite3.Row, score: float = 0.0) -> Item:
    return Item(
        id=row["id"], category=row["category"], kind=row["kind"], content=row["content"],
        status=row["status"], origin=row["origin"], confidence=row["confidence"],
        priority=row["priority"], created_at=row["created_at"], updated_at=row["updated_at"],
        access_count=row["access_count"], repeat_count=row["repeat_count"],
        times_explored=row["times_explored"], last_explored_cycle=row["last_explored_cycle"],
        meta=json.loads(row["meta"] or "{}"), score=score,
    )


class MemoryStore:
    def __init__(self, path: str | Path, clock: Any = time.time) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False,
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(_SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------ helpers
    def _x(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def transaction(self) -> "_Tx":
        return _Tx(self)

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._x("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        self._x("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))

    # -------------------------------------------------------------- items
    def get(self, item_id: int, touch: bool = False) -> Item | None:
        row = self._x("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if row and touch:
            self._x("UPDATE items SET access_count=access_count+1, last_accessed=? WHERE id=?",
                    (self.clock(), item_id))
        return _row_item(row) if row else None

    def find_exact(self, category: str, content: str) -> Item | None:
        row = self._x("SELECT * FROM items WHERE category=? AND norm_hash=? AND status NOT IN ('deleted') "
                      "ORDER BY id LIMIT 1", (category, norm_hash(content))).fetchone()
        return _row_item(row) if row else None

    def add(self, category: str, kind: str, content: str, origin: str, *,
            status: str | None = None, confidence: float | None = None, priority: float = 0.5,
            meta: dict[str, Any] | None = None, operation_id: int | None = None,
            source_ref: str | None = None, backend: str | None = None,
            model_id: str | None = None, note: str | None = None, dedupe: bool = True) -> tuple[int, bool]:
        """Insert an item. Returns (id, created). Exact duplicates are merged."""
        if category not in CATEGORIES:
            raise ValueError(f"bad category {category}")
        if origin not in ORIGINS:
            raise ValueError(f"bad origin {origin}")
        content = content.strip()
        if not content:
            raise ValueError("empty content")
        now = self.clock()
        with self.transaction():
            if dedupe and category != "episodic":
                existing = self.find_exact(category, content)
                if existing:
                    self._x("UPDATE items SET repeat_count=repeat_count+1, updated_at=? WHERE id=?",
                            (now, existing.id))
                    self.add_provenance(existing.id, origin, source_ref, operation_id, backend,
                                        model_id, note or "duplicate merged")
                    return existing.id, False
            if status is None:
                status = "open" if category == "unresolved" else "active"
            cur = self._x(
                "INSERT INTO items(category, kind, content, norm_hash, status, origin, confidence, priority,"
                " created_at, updated_at, meta) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (category, kind, content, norm_hash(content), status, origin, confidence, priority,
                 now, now, json.dumps(meta or {})),
            )
            item_id = int(cur.lastrowid)
            self.add_provenance(item_id, origin, source_ref, operation_id, backend, model_id, note)
        return item_id, True

    def add_provenance(self, item_id: int, source_type: str, source_ref: str | None = None,
                       operation_id: int | None = None, backend: str | None = None,
                       model_id: str | None = None, note: str | None = None) -> None:
        self._x("INSERT INTO provenance(item_id, source_type, source_ref, operation_id, backend, model_id,"
                " note, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (item_id, source_type, source_ref, operation_id, backend, model_id, note, self.clock()))

    def provenance(self, item_id: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._x("SELECT * FROM provenance WHERE item_id=? ORDER BY id", (item_id,))]

    def revise(self, item_id: int, reason: str, operation_id: int | None = None,
               **changes: Any) -> None:
        """Change fields of an item and record each change in the revision history."""
        allowed = {"content", "status", "confidence", "priority", "kind", "meta"}
        item = self.get(item_id)
        if item is None:
            raise KeyError(item_id)
        now = self.clock()
        with self.transaction():
            for fld, new in changes.items():
                if fld not in allowed:
                    raise ValueError(f"cannot revise {fld}")
                old = getattr(item, fld)
                if old == new:
                    continue
                stored = json.dumps(new) if fld == "meta" else new
                if fld == "content":
                    self._x("UPDATE items SET content=?, norm_hash=?, updated_at=? WHERE id=?",
                            (new, norm_hash(new), now, item_id))
                else:
                    self._x(f"UPDATE items SET {fld}=?, updated_at=? WHERE id=?", (stored, now, item_id))
                self._x("INSERT INTO revisions(item_id, field, old_value, new_value, reason, operation_id,"
                        " created_at) VALUES(?,?,?,?,?,?,?)",
                        (item_id, fld, json.dumps(old), json.dumps(new), reason, operation_id, now))

    def revisions(self, item_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if item_id is None:
            rows = self._x("SELECT * FROM revisions ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._x("SELECT * FROM revisions WHERE item_id=? ORDER BY id", (item_id,))
        return [dict(r) for r in rows]

    def mark_explored(self, item_id: int, cycle: int) -> None:
        self._x("UPDATE items SET times_explored=times_explored+1, last_explored_cycle=?, last_accessed=?"
                " WHERE id=?", (cycle, self.clock(), item_id))

    def link(self, src: int, dst: int, relation: str, weight: float = 1.0,
             operation_id: int | None = None) -> None:
        self._x("INSERT OR IGNORE INTO links(src, dst, relation, weight, operation_id, created_at)"
                " VALUES(?,?,?,?,?,?)", (src, dst, relation, weight, operation_id, self.clock()))

    def links(self, item_id: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._x(
            "SELECT * FROM links WHERE src=? OR dst=? ORDER BY id", (item_id, item_id))]

    def linked_ids(self, item_id: int, relation: str | None = None) -> set[int]:
        sql = "SELECT src, dst FROM links WHERE (src=? OR dst=?)"
        params: list[Any] = [item_id, item_id]
        if relation:
            sql += " AND relation=?"
            params.append(relation)
        out = set()
        for r in self._x(sql, params):
            out.add(r["dst"] if r["src"] == item_id else r["src"])
        return out

    def list(self, category: str | None = None, status: str | Iterable[str] | None = None,
             kind: str | None = None, limit: int = 100, order: str = "id DESC") -> list[Item]:
        sql, params = "SELECT * FROM items WHERE status != 'deleted'", []
        if category:
            sql += " AND category=?"
            params.append(category)
        if status:
            sts = [status] if isinstance(status, str) else list(status)
            sql += f" AND status IN ({','.join('?' * len(sts))})"
            params.extend(sts)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        if order not in {"id DESC", "id", "priority DESC", "updated_at DESC", "random()"}:
            raise ValueError("bad order")
        sql += f" ORDER BY {order} LIMIT ?"
        params.append(limit)
        return [_row_item(r) for r in self._x(sql, params)]

    def count(self, category: str | None = None, status: str | None = None) -> int:
        sql, params = "SELECT COUNT(*) FROM items WHERE status != 'deleted'", []
        if category:
            sql += " AND category=?"
            params.append(category)
        if status:
            sql += " AND status=?"
            params.append(status)
        return int(self._x(sql, params).fetchone()[0])

    def search(self, text: str, limit: int = 8, categories: Iterable[str] | None = None,
               exclude: Iterable[int] = (), statuses: Iterable[str] = ("active", "open", "deferred"),
               touch: bool = False) -> list[Item]:
        """Full-text retrieval ranked by BM25 with mild priority and recency terms."""
        q = fts_query(text)
        if not q:
            return []
        cats = list(categories or CATEGORIES)
        sts = list(statuses)
        excl = list(exclude)
        sql = (
            "SELECT items.*, bm25(items_fts) AS rank FROM items_fts JOIN items ON items.id = items_fts.rowid"
            " WHERE items_fts MATCH ?"
            f" AND items.category IN ({','.join('?' * len(cats))})"
            f" AND items.status IN ({','.join('?' * len(sts))})"
        )
        params: list[Any] = [q, *cats, *sts]
        if excl:
            sql += f" AND items.id NOT IN ({','.join('?' * len(excl))})"
            params.extend(excl)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit * 3)
        rows = self._x(sql, params).fetchall()
        now = self.clock()
        scored = []
        for r in rows:
            age_days = max(0.0, (now - r["updated_at"]) / 86400)
            s = -r["rank"] + 0.3 * r["priority"] + 0.2 / (1 + age_days)
            scored.append(_row_item(r, s))
        scored.sort(key=lambda it: it.score, reverse=True)
        out = scored[:limit]
        if touch and out:
            self._x(f"UPDATE items SET access_count=access_count+1, last_accessed=? WHERE id IN "
                    f"({','.join('?' * len(out))})", (now, *[i.id for i in out]))
        return out

    def near_duplicates(self, content: str, category: str | None = None, threshold: float = 0.8,
                        limit: int = 20) -> list[tuple[Item, float]]:
        words = content_words(content)
        cats = [category] if category else None
        out = []
        for it in self.search(content, limit=limit, categories=cats,
                              statuses=("active", "open", "deferred", "resolved", "superseded")):
            sim = jaccard(words, content_words(it.content))
            if sim >= threshold:
                out.append((it, sim))
        return sorted(out, key=lambda x: -x[1])

    def forget(self, item_id: int, reason: str = "user request", hard: bool = True) -> bool:
        """Delete an item. Hard deletion removes text, links and provenance."""
        item = self.get(item_id)
        if item is None:
            return False
        if hard:
            self._x("DELETE FROM items WHERE id=?", (item_id,))
        else:
            self.revise(item_id, reason, status="deleted")
        self.event("memory", "forget", {"id": item_id, "hard": hard, "reason": reason})
        return True

    # ---------------------------------------------------- process history
    def start_cycle(self, policy: str, sim_time: float | None = None) -> int:
        cur = self._x("INSERT INTO cycles(started_at, sim_time, policy, status) VALUES(?,?,?, 'running')",
                      (self.clock(), sim_time, policy))
        return int(cur.lastrowid)

    def end_cycle(self, cycle_id: int, **fields: Any) -> None:
        fields.setdefault("ended_at", self.clock())
        if "detail" in fields:
            fields["detail"] = json.dumps(fields["detail"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE cycles SET {cols} WHERE id=?", (*fields.values(), cycle_id))

    def cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]

    def start_op(self, kind: str, cycle_id: int | None = None, inp: dict[str, Any] | None = None) -> int:
        cur = self._x("INSERT INTO operations(cycle_id, kind, status, started_at, input) VALUES(?,?, 'running', ?, ?)",
                      (cycle_id, kind, self.clock(), json.dumps(inp or {}, default=str)))
        return int(cur.lastrowid)

    def end_op(self, op_id: int, status: str, out: dict[str, Any] | None = None,
               tokens_in: int = 0, tokens_out: int = 0, state_changes: int = 0,
               error: str | None = None) -> None:
        row = self._x("SELECT started_at FROM operations WHERE id=?", (op_id,)).fetchone()
        now = self.clock()
        self._x("UPDATE operations SET status=?, ended_at=?, duration_s=?, output=?, tokens_in=?, tokens_out=?,"
                " state_changes=?, error=? WHERE id=?",
                (status, now, now - row["started_at"] if row else None, json.dumps(out or {}, default=str),
                 tokens_in, tokens_out, state_changes, error, op_id))

    def operations(self, cycle_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if cycle_id is None:
            rows = self._x("SELECT * FROM operations ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._x("SELECT * FROM operations WHERE cycle_id=? ORDER BY id", (cycle_id,))
        return [dict(r) for r in rows]

    def recover(self) -> dict[str, int]:
        """Close records left 'running' by a crash or kill. Called at start-up."""
        now = self.clock()
        ops = self._x("UPDATE operations SET status='aborted', ended_at=?, error='process ended before completion'"
                      " WHERE status='running'", (now,)).rowcount
        cyc = self._x("UPDATE cycles SET status='aborted', ended_at=?, outcome='aborted' WHERE status='running'",
                      (now,)).rowcount
        cmds = self._x("UPDATE commands SET status='failed', result='\"daemon restarted\"', done_at=?"
                       " WHERE status='running'", (now,)).rowcount
        return {"operations": ops, "cycles": cyc, "commands": cmds}

    # ------------------------------------------------------------- events
    def event(self, channel: str, type_: str, payload: dict[str, Any] | None = None) -> int:
        if channel not in CHANNELS:
            raise ValueError(f"bad channel {channel}")
        cur = self._x("INSERT INTO events(ts, channel, type, payload) VALUES(?,?,?,?)",
                      (self.clock(), channel, type_, json.dumps(payload or {}, default=str)))
        return int(cur.lastrowid)

    def events(self, channel: str | None = None, after_id: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        if channel:
            rows = self._x("SELECT * FROM events WHERE channel=? AND id>? ORDER BY id DESC LIMIT ?",
                           (channel, after_id, limit))
        else:
            rows = self._x("SELECT * FROM events WHERE id>? ORDER BY id DESC LIMIT ?", (after_id, limit))
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in reversed(rows.fetchall())]

    # ----------------------------------------------------------- commands
    def enqueue(self, cmd: str, args: dict[str, Any] | None = None) -> int:
        cur = self._x("INSERT INTO commands(ts, cmd, args) VALUES(?,?,?)",
                      (self.clock(), cmd, json.dumps(args or {})))
        return int(cur.lastrowid)

    def pending_commands(self) -> list[dict[str, Any]]:
        rows = self._x("SELECT * FROM commands WHERE status='pending' ORDER BY id")
        return [dict(r) | {"args": json.loads(r["args"])} for r in rows]

    def finish_command(self, cmd_id: int, status: str, result: Any = None) -> None:
        self._x("UPDATE commands SET status=?, result=?, done_at=? WHERE id=?",
                (status, json.dumps(result, default=str), self.clock(), cmd_id))

    def command(self, cmd_id: int) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM commands WHERE id=?", (cmd_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    # ---------------------------------------------------------- retention
    def apply_retention(self, max_episodic: int, episodic_retention_days: float,
                        max_semantic: int, max_events: int) -> dict[str, int]:
        now = self.clock()
        removed = {"episodic_old": 0, "episodic_overflow": 0, "semantic_overflow": 0, "events": 0}
        with self.transaction():
            # Episodes that a retained item was derived from are kept.
            keep = "id NOT IN (SELECT dst FROM links WHERE relation='derived_from')"
            removed["episodic_old"] = self._x(
                f"DELETE FROM items WHERE category='episodic' AND created_at < ? AND {keep}",
                (now - episodic_retention_days * 86400,)).rowcount
            n = self.count("episodic")
            if n > max_episodic:
                removed["episodic_overflow"] = self._x(
                    f"DELETE FROM items WHERE id IN (SELECT id FROM items WHERE category='episodic' AND {keep}"
                    " ORDER BY priority ASC, id ASC LIMIT ?)", (n - max_episodic,)).rowcount
            n = self.count("semantic")
            if n > max_semantic:
                # Superseded/discarded first, then the least accessed low-confidence inferred items.
                removed["semantic_overflow"] = self._x(
                    "DELETE FROM items WHERE id IN (SELECT id FROM items WHERE category='semantic'"
                    " ORDER BY (status IN ('superseded','discarded')) DESC, (origin='inferred') DESC,"
                    " access_count ASC, COALESCE(confidence, 0) ASC, id ASC LIMIT ?)",
                    (n - max_semantic,)).rowcount
            total = int(self._x("SELECT COUNT(*) FROM events").fetchone()[0])
            if total > max_events:
                removed["events"] = self._x(
                    "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id ASC LIMIT ?)",
                    (total - max_events,)).rowcount
        return removed

    def stats(self) -> dict[str, Any]:
        rows = self._x("SELECT category, status, COUNT(*) AS n FROM items GROUP BY category, status")
        by: dict[str, dict[str, int]] = {}
        for r in rows:
            by.setdefault(r["category"], {})[r["status"]] = r["n"]
        origins = {r["origin"]: r["n"] for r in self._x(
            "SELECT origin, COUNT(*) AS n FROM items WHERE status!='deleted' GROUP BY origin")}
        size = Path(self.path).stat().st_size if self.path != ":memory:" else 0
        return {
            "items": by,
            "origins": origins,
            "links": int(self._x("SELECT COUNT(*) FROM links").fetchone()[0]),
            "revisions": int(self._x("SELECT COUNT(*) FROM revisions").fetchone()[0]),
            "cycles": int(self._x("SELECT COUNT(*) FROM cycles").fetchone()[0]),
            "operations": int(self._x("SELECT COUNT(*) FROM operations").fetchone()[0]),
            "events": int(self._x("SELECT COUNT(*) FROM events").fetchone()[0]),
            "db_bytes": size,
        }

    def export(self) -> dict[str, Any]:
        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(r) for r in self._x(sql)]
        return {
            "schema_version": SCHEMA_VERSION,
            "items": rows("SELECT * FROM items WHERE status != 'deleted' ORDER BY id"),
            "links": rows("SELECT * FROM links ORDER BY id"),
            "provenance": rows("SELECT * FROM provenance ORDER BY id"),
            "revisions": rows("SELECT * FROM revisions ORDER BY id"),
            "cycles": rows("SELECT * FROM cycles ORDER BY id"),
        }

    def snapshot(self, dest: str | Path) -> None:
        """Consistent copy of the database (used for reproducible experiment starts)."""
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(dest))
        with self._lock:
            self.conn.backup(target)
        target.close()


class _Tx:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.outer = False

    def __enter__(self) -> "_Tx":
        self.store._lock.acquire()
        self.outer = not self.store.conn.in_transaction
        if self.outer:
            self.store.conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        try:
            if self.outer:
                self.store.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.store._lock.release()
