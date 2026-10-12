"""Persistent weight experiment records (same SQLite file as memory)."""

from __future__ import annotations

import json
import time
from typing import Any

from ..memory.store import MemoryStore

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS weight_experiments (
    id INTEGER PRIMARY KEY,
    hypothesis_item_id INTEGER,
    parent_checkpoint TEXT NOT NULL,
    parent_sha256 TEXT NOT NULL,
    intervention_method TEXT NOT NULL,
    intervention_json TEXT NOT NULL,
    intervention_fingerprint TEXT NOT NULL,
    candidate_path TEXT,
    baseline_path TEXT,
    status TEXT NOT NULL,
    decision TEXT,
    decision_reason TEXT,
    created_at REAL NOT NULL,
    completed_at REAL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS weight_exp_fp ON weight_experiments(intervention_fingerprint);
CREATE INDEX IF NOT EXISTS weight_exp_hyp ON weight_experiments(hypothesis_item_id);
CREATE TABLE IF NOT EXISTS weight_measurements (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES weight_experiments(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS weight_meas_exp ON weight_measurements(experiment_id);
"""


def ensure_schema(store: MemoryStore) -> None:
    ver = store.get_meta("schema_version", 1)
    if ver >= 2:
        return
    store.conn.executescript(_SCHEMA_V2)
    store.set_meta("schema_version", 2)


def create_experiment(store: MemoryStore, *, hypothesis_item_id: int | None, parent_checkpoint: str,
                      parent_sha256: str, method: str, intervention: dict[str, Any],
                      fingerprint: str, baseline_path: str) -> int:
    ensure_schema(store)
    now = store.clock()
    cur = store._x(
        "INSERT INTO weight_experiments(hypothesis_item_id, parent_checkpoint, parent_sha256,"
        " intervention_method, intervention_json, intervention_fingerprint, baseline_path, status, created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (hypothesis_item_id, parent_checkpoint, parent_sha256, method, json.dumps(intervention),
         fingerprint, baseline_path, "running", now),
    )
    return int(cur.lastrowid)


def complete_experiment(store: MemoryStore, exp_id: int, *, decision: str, reason: str,
                        candidate_path: str, detail: dict[str, Any]) -> None:
    store._x(
        "UPDATE weight_experiments SET status='completed', decision=?, decision_reason=?,"
        " candidate_path=?, completed_at=?, detail=? WHERE id=?",
        (decision, reason, candidate_path, store.clock(), json.dumps(detail, default=str), exp_id),
    )


def fail_experiment(store: MemoryStore, exp_id: int, error: str) -> None:
    store._x(
        "UPDATE weight_experiments SET status='failed', decision='failed', decision_reason=?, completed_at=?"
        " WHERE id=?",
        (error, store.clock(), exp_id),
    )


def record_measurement(store: MemoryStore, exp_id: int, phase: str, metric: str, value: float,
                       detail: dict[str, Any] | None = None) -> None:
    store._x(
        "INSERT INTO weight_measurements(experiment_id, phase, metric, value, detail) VALUES(?,?,?,?,?)",
        (exp_id, phase, metric, value, json.dumps(detail or {}, default=str)),
    )


def fingerprint_blocked(store: MemoryStore, fingerprint: str) -> bool:
    """True if this intervention was already rejected or failed."""
    ensure_schema(store)
    row = store._x(
        "SELECT decision FROM weight_experiments WHERE intervention_fingerprint=? AND status='completed'"
        " AND decision IN ('reject','failed','regressed') ORDER BY id DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row is not None


def list_experiments(store: MemoryStore, limit: int = 20) -> list[dict[str, Any]]:
    ensure_schema(store)
    rows = store._x("SELECT * FROM weight_experiments ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]


def pending_weight_hypotheses(store: MemoryStore, limit: int = 10) -> list[Any]:
    """Open weight hypotheses with no completed experiment yet."""
    ensure_schema(store)
    items = store.list("unresolved", "open", kind="weight_hypothesis", limit=200)
    out = []
    for it in items:
        accepted = store._x(
            "SELECT id FROM weight_experiments WHERE hypothesis_item_id=? AND status='completed'"
            " AND decision='accept' LIMIT 1",
            (it.id,),
        ).fetchone()
        if not accepted:
            out.append(it)
        if len(out) >= limit:
            break
    return out
