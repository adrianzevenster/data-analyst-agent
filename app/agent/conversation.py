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
    groundedness_score: int | None = None
    groundedness_criteria: dict[str, int] = field(default_factory=dict)
    groundedness_issues: list[str] = field(default_factory=list)
    judge_status: str = "rule_based"
    planning_source: str = "rules"
    synthesis_source: str = "rules"


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
                    "groundedness_score": t.groundedness_score,
                    "groundedness_criteria": t.groundedness_criteria,
                    "groundedness_issues": t.groundedness_issues,
                    "judge_status": t.judge_status,
                    "planning_source": t.planning_source,
                    "synthesis_source": t.synthesis_source,
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
                groundedness_score=t.get("groundedness_score"),
                groundedness_criteria=t.get("groundedness_criteria", {}),
                groundedness_issues=t.get("groundedness_issues", []),
                judge_status=t.get("judge_status", "rule_based"),
                planning_source=t.get("planning_source", "rules"),
                synthesis_source=t.get("synthesis_source", "rules"),
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
