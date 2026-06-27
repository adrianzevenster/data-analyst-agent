from __future__ import annotations

import pandas as pd
import pytest

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager
from app.agent.llm import _tools_run_recently


def test_tools_run_recently_empty_history():
    assert _tools_run_recently(None) == set()
    assert _tools_run_recently([]) == set()


def test_tools_run_recently_ignores_user_turns():
    history = [
        {"role": "user", "content": "hi", "tool_results": [{"tool": "profile_dataset"}]},
    ]
    assert _tools_run_recently(history) == set()


def test_tools_run_recently_collects_from_assistant_turns():
    history = [
        {"role": "user", "content": "analyse"},
        {"role": "assistant", "content": "ok", "tool_results": [
            {"tool": "profile_dataset"},
            {"tool": "auto_insights"},
        ]},
    ]
    assert _tools_run_recently(history) == {"profile_dataset", "auto_insights"}


def test_tools_run_recently_respects_last_n_turns():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "old", "tool_results": [{"tool": "trend_analysis"}]},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "recent", "tool_results": [{"tool": "auto_insights"}]},
    ]
    # last_n_turns=1 should only see the most recent assistant turn
    result = _tools_run_recently(history, last_n_turns=1)
    assert result == {"auto_insights"}
    assert "trend_analysis" not in result


def test_tools_run_recently_all_within_window():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "t1", "tool_results": [{"tool": "trend_analysis"}]},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "t2", "tool_results": [{"tool": "auto_insights"}]},
    ]
    result = _tools_run_recently(history, last_n_turns=3)
    assert result == {"trend_analysis", "auto_insights"}


def test_rule_planner_no_repeat_eda_after_recent_run(tmp_path, monkeypatch):
    """Rule planner must not fall back to auto_insights when it ran recently."""
    monkeypatch.setenv("ENABLE_RAG", "0")
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "small.csv"
    )
    planner = Planner()
    planner.dm = manager

    history = [
        {"role": "user", "content": "what's interesting?"},
        {"role": "assistant", "content": "Here are some insights...", "tool_results": [
            {"tool": "auto_insights"},
        ]},
    ]
    # A vague follow-up message that would otherwise trigger the auto_insights fallback.
    calls, *_ = planner.plan("tell me more", meta.dataset_id, conversation_history=history)
    names = [c.name for c in calls]
    assert "auto_insights" not in names


def test_rule_planner_selects_ml_evaluation_for_prediction_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")

    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame(
            {
                "actual": [1, 0, 1],
                "prediction": [1, 1, 1],
                "probability": [0.9, 0.7, 0.6],
            }
        ),
        "predictions.csv",
    )

    planner = Planner()
    planner.dm = manager

    calls, citations, source, llm_error, llm_notes = planner.plan("evaluate model performance", meta.dataset_id)

    assert citations == []
    assert [call.name for call in calls] == ["evaluate_ml_predictions"]
    assert calls[0].arguments["task_hint"] == "classification"
