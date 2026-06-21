from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_RECENT_JUDGEMENTS = 200
LOW_SCORE_THRESHOLD = 3
JUDGE_STATUSES = ("judged", "not_sampled", "rule_based", "llm_disabled", "failed")


@dataclass
class JudgeRecord:
    score: int
    issue_count: int
    timestamp: float = field(default_factory=time.time)


class _JudgeStore:
    """SQLite persistence for groundedness scores. Reuses conversations.db."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from app.core.config import settings
            db_path = str(settings.data_path / "conversations.db")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    score INTEGER NOT NULL,
                    issue_count INTEGER NOT NULL,
                    synthesis_source TEXT NOT NULL DEFAULT 'llm',
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def insert(self, rec: JudgeRecord, synthesis_source: str = "llm") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO judge_log (score, issue_count, synthesis_source, timestamp) VALUES (?, ?, ?, ?)",
                (rec.score, rec.issue_count, synthesis_source, rec.timestamp),
            )
            conn.commit()

    def recent(self, n: int = MAX_RECENT_JUDGEMENTS) -> list[JudgeRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT score, issue_count, timestamp FROM judge_log ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [JudgeRecord(score=r[0], issue_count=r[1], timestamp=r[2]) for r in rows]

    def history(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT score, issue_count, synthesis_source, timestamp "
                "FROM judge_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"score": r[0], "issue_count": r[1], "synthesis_source": r[2], "timestamp": r[3]}
            for r in rows
        ]


class JudgeMetrics:
    """Aggregates sampled LLM-as-judge groundedness scores.

    Scores are persisted to SQLite so the trend survives API restarts.
    The in-memory ring buffer keeps snapshot() fast; the store provides
    history across sessions.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[JudgeRecord] = []
        self._status_counts: dict[str, int] = {status: 0 for status in JUDGE_STATUSES}
        self._last_error: str | None = None
        try:
            self._store: _JudgeStore | None = _JudgeStore(db_path)
            with self._lock:
                self._records = self._store.recent(MAX_RECENT_JUDGEMENTS)
                self._status_counts["judged"] = len(self._records)
        except Exception:
            self._store = None

    def _mark_status_locked(self, status: str) -> None:
        if status not in self._status_counts:
            status = "failed"
        self._status_counts[status] += 1

    def record(self, rec: JudgeRecord, synthesis_source: str = "llm") -> None:
        with self._lock:
            self._mark_status_locked("judged")
            self._records.append(rec)
            if len(self._records) > MAX_RECENT_JUDGEMENTS:
                self._records = self._records[-MAX_RECENT_JUDGEMENTS:]
        if self._store is not None:
            try:
                self._store.insert(rec, synthesis_source)
            except Exception:
                pass

    def record_skipped(self, status: str) -> None:
        if status not in {"not_sampled", "rule_based", "llm_disabled"}:
            raise ValueError(f"Invalid judge skip status: {status}")
        with self._lock:
            self._mark_status_locked(status)

    def record_failure(self, error: str | None = None) -> None:
        with self._lock:
            self._mark_status_locked("failed")
            if error:
                self._last_error = error

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)
            status_counts = dict(self._status_counts)
            last_error = self._last_error

        total = len(records)
        avg_score = sum(r.score for r in records) / total if total else 0.0
        low_score_count = sum(1 for r in records if r.score <= LOW_SCORE_THRESHOLD)
        flagged_count = sum(1 for r in records if r.issue_count > 0)
        skipped_sample_rate_count = status_counts["not_sampled"]
        skipped_rule_based_count = status_counts["rule_based"]
        skipped_llm_disabled_count = status_counts["llm_disabled"]
        error_count = status_counts["failed"]

        return {
            "response_count": sum(status_counts.values()),
            "eligible_count": status_counts["judged"] + skipped_sample_rate_count + error_count,
            "attempted_count": status_counts["judged"] + error_count,
            "sampled_count": total,
            "skipped_count": (
                skipped_sample_rate_count
                + skipped_rule_based_count
                + skipped_llm_disabled_count
            ),
            "skipped_sample_rate_count": skipped_sample_rate_count,
            "skipped_rule_based_count": skipped_rule_based_count,
            "skipped_llm_disabled_count": skipped_llm_disabled_count,
            "error_count": error_count,
            "avg_groundedness_score": round(avg_score, 2),
            "low_score_rate": round(low_score_count / total, 4) if total else 0.0,
            "flagged_rate": round(flagged_count / total, 4) if total else 0.0,
            "last_error": last_error,
        }

    def history(self, limit: int = 500) -> list[dict]:
        if self._store is not None:
            try:
                return self._store.history(limit)
            except Exception:
                pass
        with self._lock:
            records = list(self._records[-limit:])
        return [
            {"score": r.score, "issue_count": r.issue_count, "synthesis_source": "llm", "timestamp": r.timestamp}
            for r in reversed(records)
        ]


judge_metrics = JudgeMetrics()
