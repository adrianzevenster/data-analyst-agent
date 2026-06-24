from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCORE_MIN = 1
_SCORE_MAX = 5


class QualityEvalPipeline:
    """Offline/scheduled quality evaluation over stored conversation turns.

    Complements the request-time LLM judge by sampling turns that were
    previously rule-based or not_sampled and judging them after the fact.
    Results are written into the shared `judge_log` table so they appear
    in the quality trend alongside real-time scores.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from app.core.config import settings
            db_path = str(settings.data_path / "conversations.db")
        self._db = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    def _init_table(self) -> None:
        with self._conn() as conn:
            # Conversations and judge_log already exist in the production DB;
            # create them here so the pipeline is self-contained on fresh test DBs.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    score INTEGER NOT NULL,
                    issue_count INTEGER NOT NULL,
                    synthesis_source TEXT NOT NULL DEFAULT 'llm',
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_run_log (
                    run_id TEXT PRIMARY KEY,
                    n_sampled INTEGER NOT NULL,
                    n_judged INTEGER NOT NULL,
                    n_failed INTEGER NOT NULL,
                    avg_score REAL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    # ── Quality trend ─────────────────────────────────────────────────────────

    def quality_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Return per-day aggregate scores from judge_log, newest-first."""
        cutoff = time.time() - days * 86400
        with self._conn() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT
                        date(timestamp, 'unixepoch') AS day,
                        round(avg(score), 2)          AS avg_score,
                        count(*)                      AS n,
                        min(score)                    AS min_score,
                        max(score)                    AS max_score
                    FROM judge_log
                    WHERE timestamp > ?
                    GROUP BY day
                    ORDER BY day ASC
                    """,
                    (cutoff,),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [
            {"day": r[0], "avg_score": r[1], "n": r[2], "min_score": r[3], "max_score": r[4]}
            for r in rows
        ]

    # ── Turn sampling ─────────────────────────────────────────────────────────

    def _sample_turns(self, n: int, max_age_days: int) -> list[dict[str, Any]]:
        """Pull up to `n` recent assistant turns from conversation blobs."""
        cutoff = time.time() - max_age_days * 86400
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM conversations WHERE updated_at > ? ORDER BY updated_at DESC LIMIT 200",
                (cutoff,),
            ).fetchall()

        candidates: list[dict[str, Any]] = []
        for (blob,) in rows:
            try:
                conv = json.loads(blob)
            except Exception:
                continue
            conv_id = conv.get("conversation_id", "unknown")
            for turn in conv.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                if not turn.get("content", "").strip():
                    continue
                candidates.append({
                    "conversation_id": conv_id,
                    "content": turn["content"],
                    "dataset_id": turn.get("dataset_id"),
                    "tool_results": turn.get("tool_results", []),
                    "timestamp": turn.get("timestamp", 0.0),
                })
            if len(candidates) >= n * 4:
                break

        # Sort by timestamp descending, pick the freshest n
        candidates.sort(key=lambda t: t["timestamp"], reverse=True)
        return candidates[:n]

    # ── Eval run ──────────────────────────────────────────────────────────────

    def run(
        self,
        reasoner: Any,
        dm: Any,
        n: int = 20,
        max_age_days: int = 7,
    ) -> dict[str, Any]:
        """Sample turns, judge each with the LLM, persist results.

        Returns a summary dict: run_id, n_sampled, n_judged, n_failed, avg_score.
        No-ops gracefully when the LLM is unavailable (returns n_judged=0).
        """
        run_id = str(uuid.uuid4())[:8]
        turns = self._sample_turns(n, max_age_days)
        n_sampled = len(turns)
        scores: list[int] = []
        n_failed = 0

        for turn in turns:
            dataset_context = None
            if turn.get("dataset_id"):
                try:
                    from app.core.config import settings
                    df = dm.load_df(turn["dataset_id"], settings.llm_analysis_sample_rows)
                    dataset_context = reasoner.dataset_analysis_context(df)
                except Exception:
                    pass

            try:
                verdict = reasoner.judge_groundedness(
                    turn["content"],
                    dataset_context=dataset_context,
                    tool_results=turn["tool_results"],
                )
                score = int(verdict["score"])
                issue_count = len(verdict.get("issues", []))
                scores.append(score)
                # Write into shared judge_log so quality_trend picks it up
                with self._conn() as conn:
                    conn.execute(
                        "INSERT INTO judge_log (score, issue_count, synthesis_source, timestamp) VALUES (?, ?, ?, ?)",
                        (score, issue_count, "eval_run", time.time()),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning("eval_run: judge failed for turn: %s", e)
                n_failed += 1

        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        n_judged = len(scores)

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO eval_run_log (run_id, n_sampled, n_judged, n_failed, avg_score, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, n_sampled, n_judged, n_failed, avg_score, time.time()),
            )
            conn.commit()

        return {
            "run_id": run_id,
            "n_sampled": n_sampled,
            "n_judged": n_judged,
            "n_failed": n_failed,
            "avg_score": avg_score,
        }

    def run_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return most recent eval-run summaries."""
        with self._conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT run_id, n_sampled, n_judged, n_failed, avg_score, timestamp "
                    "FROM eval_run_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [
            {
                "run_id": r[0],
                "n_sampled": r[1],
                "n_judged": r[2],
                "n_failed": r[3],
                "avg_score": r[4],
                "timestamp": r[5],
            }
            for r in rows
        ]
