from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from app.analytics.quality_eval import QualityEvalPipeline


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def pipeline(tmp_path):
    return QualityEvalPipeline(db_path=str(tmp_path / "test.db"))


def _seed_judge_log(db_path: str, rows: list[tuple[int, int, str, float]]) -> None:
    """Insert (score, issue_count, synthesis_source, timestamp) rows into judge_log."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS judge_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score INTEGER NOT NULL,
            issue_count INTEGER NOT NULL,
            synthesis_source TEXT NOT NULL DEFAULT 'llm',
            timestamp REAL NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO judge_log (score, issue_count, synthesis_source, timestamp) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _seed_conversations(db_path: str, turns: list[dict]) -> None:
    """Insert a single conversation blob with the given assistant turns."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    data = json.dumps({
        "conversation_id": "test-conv",
        "last_dataset_id": None,
        "trained_model_ids": [],
        "turns": turns,
    })
    conn.execute(
        "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?)",
        ("test-conv", data, time.time()),
    )
    conn.commit()
    conn.close()


# ── quality_trend ─────────────────────────────────────────────────────────────

def test_quality_trend_empty_when_no_data(pipeline):
    result = pipeline.quality_trend(days=30)
    assert result == []


def test_quality_trend_excludes_old_entries(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    old_ts = time.time() - 40 * 86400  # 40 days ago
    recent_ts = time.time() - 1 * 86400  # yesterday
    _seed_judge_log(db, [
        (5, 0, "llm", old_ts),
        (3, 1, "llm", recent_ts),
    ])
    rows = pipeline.quality_trend(days=30)
    assert len(rows) == 1
    assert rows[0]["avg_score"] == 3.0


def test_quality_trend_aggregates_per_day(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    ts_today = time.time()
    _seed_judge_log(db, [
        (4, 0, "llm", ts_today),
        (2, 1, "llm", ts_today),
    ])
    rows = pipeline.quality_trend(days=7)
    assert len(rows) == 1
    assert rows[0]["avg_score"] == 3.0
    assert rows[0]["n"] == 2
    assert rows[0]["min_score"] == 2
    assert rows[0]["max_score"] == 4


# ── sample_turns ──────────────────────────────────────────────────────────────

def test_sample_turns_returns_assistant_turns_only(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    _seed_conversations(db, [
        {"role": "user", "content": "query", "timestamp": time.time()},
        {"role": "assistant", "content": "answer", "timestamp": time.time()},
    ])
    turns = pipeline._sample_turns(n=10, max_age_days=7)
    assert all(t.get("content") == "answer" for t in turns)
    assert len(turns) == 1


def test_sample_turns_empty_when_no_conversations(pipeline):
    assert pipeline._sample_turns(n=10, max_age_days=7) == []


def test_sample_turns_respects_n_cap(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    turns_data = [
        {"role": "assistant", "content": f"answer {i}", "timestamp": time.time() - i}
        for i in range(10)
    ]
    _seed_conversations(db, turns_data)
    sampled = pipeline._sample_turns(n=3, max_age_days=7)
    assert len(sampled) <= 3


# ── eval run ──────────────────────────────────────────────────────────────────

def test_run_returns_zero_judged_when_no_turns(pipeline):
    reasoner = MagicMock()
    dm = MagicMock()
    result = pipeline.run(reasoner=reasoner, dm=dm, n=10, max_age_days=7)
    assert result["n_sampled"] == 0
    assert result["n_judged"] == 0
    assert result["avg_score"] is None
    assert "run_id" in result


def test_run_judges_available_turns(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    _seed_conversations(db, [
        {"role": "assistant", "content": "good analysis", "timestamp": time.time(), "tool_results": []},
    ])

    reasoner = MagicMock()
    reasoner.judge_groundedness.return_value = {"score": 4, "issues": []}
    reasoner.dataset_analysis_context.return_value = None
    dm = MagicMock()

    result = pipeline.run(reasoner=reasoner, dm=dm, n=10, max_age_days=7)
    assert result["n_judged"] == 1
    assert result["avg_score"] == 4.0
    assert result["n_failed"] == 0


def test_run_handles_judge_failure_gracefully(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    _seed_conversations(db, [
        {"role": "assistant", "content": "answer", "timestamp": time.time(), "tool_results": []},
    ])

    reasoner = MagicMock()
    reasoner.judge_groundedness.side_effect = RuntimeError("LLM timeout")
    dm = MagicMock()

    result = pipeline.run(reasoner=reasoner, dm=dm, n=10, max_age_days=7)
    assert result["n_judged"] == 0
    assert result["n_failed"] == 1
    assert result["avg_score"] is None


def test_run_writes_scores_to_judge_log(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    _seed_conversations(db, [
        {"role": "assistant", "content": "insight", "timestamp": time.time(), "tool_results": []},
    ])

    reasoner = MagicMock()
    reasoner.judge_groundedness.return_value = {"score": 5, "issues": []}
    dm = MagicMock()

    pipeline.run(reasoner=reasoner, dm=dm, n=10, max_age_days=7)

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT score, synthesis_source FROM judge_log").fetchall()
    conn.close()
    assert any(r[0] == 5 and r[1] == "eval_run" for r in rows)


def test_run_history_records_each_run(tmp_path):
    db = str(tmp_path / "test.db")
    pipeline = QualityEvalPipeline(db_path=db)
    reasoner = MagicMock()
    dm = MagicMock()

    pipeline.run(reasoner=reasoner, dm=dm, n=5, max_age_days=7)
    pipeline.run(reasoner=reasoner, dm=dm, n=5, max_age_days=7)

    history = pipeline.run_history()
    assert len(history) == 2
    assert all("run_id" in h for h in history)
