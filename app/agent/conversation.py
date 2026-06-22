from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings

MAX_TURNS = 20
DEFAULT_HISTORY_TURNS = 6
CONVERSATION_TTL_DAYS = 7

# Fields that are too large or uninformative to include in the planning context.
_HISTORY_SKIP_FIELDS = frozenset({
    "engineering_readout", "readout", "charts", "columns", "rows",
    "scored_rows", "eval_df", "feature_importances", "feature_importance",
    "preprocessing_notes", "leakage_warnings",
})
_HISTORY_SCALAR_TYPES = (int, float, str)


def _slim_tool_results_for_history(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress stored tool-result dicts for inclusion in planning context.

    Keeps: tool name, ok flag, engineering_readout (as "summary"), scalar top-level
    metrics, and short text lists (findings, notes). Drops: charts, tables, nested
    dicts, large arrays. Caps total entries at 8 to limit prompt size.
    """
    slim = []
    for tr in tool_results[:8]:
        entry: dict[str, Any] = {"tool": tr.get("name", "unknown"), "ok": tr.get("ok", False)}
        if not tr.get("ok"):
            if tr.get("error"):
                entry["error"] = str(tr["error"])[:200]
        else:
            result = tr.get("result") or {}
            if isinstance(result, dict):
                readout = result.get("engineering_readout") or result.get("readout")
                if readout:
                    entry["summary"] = str(readout)[:400]
                for key in ("findings", "notes", "warnings"):
                    val = result.get(key)
                    if isinstance(val, list) and val and isinstance(val[0], str):
                        entry[key] = val[:5]
                for key, val in result.items():
                    if key in _HISTORY_SKIP_FIELDS:
                        continue
                    if isinstance(val, _HISTORY_SCALAR_TYPES) and not isinstance(val, bool):
                        entry[key] = val
        slim.append(entry)
    return slim


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    dataset_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    # Rich response data (assistant turns only)
    tables: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    groundedness_score: int | None = None
    groundedness_criteria: dict[str, int] = field(default_factory=dict)
    groundedness_issues: list[str] = field(default_factory=list)
    judge_status: str = "rule_based"
    planning_source: str = "rules"
    synthesis_source: str = "rules"
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    last_dataset_id: str | None = None
    trained_model_ids: list[str] = field(default_factory=list)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]

    def recent_history(self, n: int = DEFAULT_HISTORY_TURNS) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns[-n:]]

    def recent_history_with_tool_context(self, n: int = DEFAULT_HISTORY_TURNS) -> list[dict[str, Any]]:
        """Like recent_history() but enriches assistant turns with a slim tool-result summary.

        The planner uses this to understand what has already been computed across
        prior turns — e.g., which model_id was trained, what drift was found —
        without receiving the full (potentially large) result payloads.
        """
        result: list[dict[str, Any]] = []
        for t in self.turns[-n:]:
            entry: dict[str, Any] = {"role": t.role, "content": t.content}
            if t.role == "assistant" and t.tool_results:
                entry["tool_results"] = _slim_tool_results_for_history(t.tool_results)
            result.append(entry)
        return result


class ConversationStore:
    """SQLite-backed conversation store. Survives API restarts.

    An in-process cache keeps live Conversation objects so repeated
    get_or_create calls within the same process return the same instance.
    """

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(settings.data_path / "conversations.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = path
        self._cache: dict[str, Conversation] = {}
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def _serialize(conv: Conversation) -> str:
        return json.dumps({
            "conversation_id": conv.conversation_id,
            "last_dataset_id": conv.last_dataset_id,
            "trained_model_ids": conv.trained_model_ids,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "dataset_id": t.dataset_id,
                    "tool_calls": t.tool_calls,
                    "timestamp": t.timestamp,
                    "tables": t.tables,
                    "charts": t.charts,
                    "tool_results": t.tool_results,
                    "groundedness_score": t.groundedness_score,
                    "groundedness_criteria": t.groundedness_criteria,
                    "groundedness_issues": t.groundedness_issues,
                    "judge_status": t.judge_status,
                    "planning_source": t.planning_source,
                    "synthesis_source": t.synthesis_source,
                    "latency_ms": t.latency_ms,
                }
                for t in conv.turns
            ],
        })

    @staticmethod
    def _deserialize(data: str) -> Conversation:
        d = json.loads(data)
        conv = Conversation(
            conversation_id=d["conversation_id"],
            last_dataset_id=d.get("last_dataset_id"),
            trained_model_ids=d.get("trained_model_ids", []),
        )
        conv.turns = [
            Turn(
                role=t["role"],
                content=t["content"],
                dataset_id=t.get("dataset_id"),
                tool_calls=t.get("tool_calls", []),
                timestamp=t.get("timestamp", 0.0),
                tables=t.get("tables", []),
                charts=t.get("charts", []),
                tool_results=t.get("tool_results", []),
                groundedness_score=t.get("groundedness_score"),
                groundedness_criteria=t.get("groundedness_criteria", {}),
                groundedness_issues=t.get("groundedness_issues", []),
                judge_status=t.get("judge_status", "rule_based"),
                planning_source=t.get("planning_source", "rules"),
                synthesis_source=t.get("synthesis_source", "rules"),
                latency_ms=t.get("latency_ms", {}),
            )
            for t in d.get("turns", [])
        ]
        return conv

    def get_or_create(self, conversation_id: str) -> Conversation:
        if conversation_id in self._cache:
            return self._cache[conversation_id]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        conv = self._deserialize(row[0]) if row else Conversation(conversation_id=conversation_id)
        self._cache[conversation_id] = conv
        return conv

    def save(self, conv: Conversation) -> None:
        self._cache[conv.conversation_id] = conv
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (conversation_id, data, updated_at) VALUES (?, ?, ?)",
                (conv.conversation_id, self._serialize(conv), time.time()),
            )
            conn.commit()

    def evict_old(self) -> None:
        cutoff = time.time() - CONVERSATION_TTL_DAYS * 86400
        with self._conn() as conn:
            conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))
            conn.commit()
