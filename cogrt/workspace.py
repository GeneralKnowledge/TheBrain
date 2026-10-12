"""Temporary, structured working state.

The workspace is distinct from durable memory: it holds the current focus,
retrieved context, candidate updates and the reasons they were kept or
dropped. Every slot is bounded. A JSON snapshot is saved after each stage so
that a restarted process resumes with the same focus, but nothing in the
workspace counts as a retained memory until the commit stage writes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SLOTS = ("hypotheses", "questions", "contradictions", "conclusions", "relevant",
         "associations", "proposals", "decisions", "recent_topics")


@dataclass
class Focus:
    item_id: int | None = None
    policy: str | None = None
    stage: str | None = None
    cycle_id: int | None = None
    content: str = ""
    secondary_id: int | None = None


@dataclass
class Workspace:
    max_items: int = 8
    max_topics: int = 16
    focus: Focus = field(default_factory=Focus)
    slots: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {s: [] for s in SLOTS})
    pending_user: list[dict[str, Any]] = field(default_factory=list)

    def push(self, slot: str, entry: dict[str, Any]) -> None:
        lst = self.slots[slot]
        key = entry.get("id")
        if key is not None:
            lst[:] = [e for e in lst if e.get("id") != key]
        lst.append(entry)
        limit = self.max_topics if slot == "recent_topics" else self.max_items
        del lst[:-limit]

    def set(self, slot: str, entries: list[dict[str, Any]]) -> None:
        limit = self.max_topics if slot == "recent_topics" else self.max_items
        self.slots[slot] = list(entries)[-limit:]

    def clear(self, *slots: str) -> None:
        for s in slots:
            self.slots[s] = []

    def recently_explored(self, item_id: int) -> bool:
        return any(e.get("id") == item_id for e in self.slots["recent_topics"])

    def refresh(self, store: Any) -> None:
        """Mirror the top of durable state into the visible slots."""
        def brief(items: list[Any]) -> list[dict[str, Any]]:
            return [{"id": i.id, "content": i.content[:200], "confidence": i.confidence,
                     "origin": i.origin} for i in items]
        self.set("hypotheses", brief(store.list("unresolved", "open", kind="hypothesis",
                                                limit=self.max_items, order="priority DESC")))
        self.set("questions", brief(store.list("unresolved", "open", kind="question",
                                               limit=self.max_items, order="priority DESC")))
        self.set("contradictions", brief(store.list("unresolved", "open", kind="contradiction",
                                                    limit=self.max_items, order="priority DESC")))
        self.set("conclusions", brief(store.list("semantic", "active", limit=self.max_items,
                                                 order="updated_at DESC")))

    def to_dict(self) -> dict[str, Any]:
        return {"focus": asdict(self.focus), "slots": self.slots, "pending_user": self.pending_user}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any], max_items: int = 8, max_topics: int = 16) -> "Workspace":
        ws = cls(max_items=max_items, max_topics=max_topics)
        ws.focus = Focus(**data.get("focus", {}))
        for s in SLOTS:
            ws.slots[s] = list(data.get("slots", {}).get(s, []))
        ws.pending_user = list(data.get("pending_user", []))
        return ws

    def size(self) -> int:
        return sum(len(v) for v in self.slots.values())
