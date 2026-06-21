from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ExperimentTracker:
    """Persists every training run to an `experiments` table in conversations.db.

    Each row captures the full context of one train_supervised_model call —
    hyperparams, metrics, preprocessing decisions — so runs can be queried
    and compared without re-running anything.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from app.core.config import settings
            db_path = str(settings.data_path / "conversations.db")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = db_path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id      TEXT PRIMARY KEY,
                    model_id    TEXT NOT NULL,
                    dataset_id  TEXT,
                    target_col  TEXT NOT NULL,
                    task_type   TEXT NOT NULL,
                    model_type  TEXT NOT NULL,
                    params      TEXT NOT NULL,
                    metrics     TEXT NOT NULL,
                    preprocessing TEXT NOT NULL,
                    comparison  TEXT,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_run(
        self,
        *,
        model_id: str,
        dataset_id: str | None,
        target_col: str,
        task_type: str,
        model_type: str,
        params: dict,
        metrics: dict,
        preprocessing: dict,
        comparison: dict | None = None,
    ) -> str:
        run_id = model_id  # 1-to-1 with model registry for easy cross-reference
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (run_id, model_id, dataset_id, target_col, task_type, model_type,
                    params, metrics, preprocessing, comparison, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, model_id, dataset_id, target_col, task_type, model_type,
                    json.dumps(params),
                    json.dumps(metrics),
                    json.dumps(preprocessing),
                    json.dumps(comparison) if comparison else None,
                    now,
                ),
            )
            conn.commit()
        return run_id

    def list_runs(
        self,
        dataset_id: str | None = None,
        target_col: str | None = None,
        model_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses: list[str] = []
        args: list[object] = []
        if dataset_id is not None:
            clauses.append("dataset_id = ?")
            args.append(dataset_id)
        if target_col is not None:
            clauses.append("target_col = ?")
            args.append(target_col)
        if model_type is not None:
            clauses.append("model_type = ?")
            args.append(model_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?",
                args,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("params", "metrics", "preprocessing", "comparison"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d


# Module-level singleton — shared across training and API routes.
_tracker: ExperimentTracker | None = None


def get_tracker(db_path: str | None = None) -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(db_path)
    return _tracker
