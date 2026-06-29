from __future__ import annotations

import pandas as pd

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


def test_planner_chains_train_and_explain_with_sentinel():
    import numpy as np
    from app.agent.llm import LATEST_TRAINED_MODEL_SENTINEL
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    calls = planner._rule_plan(
        "Train a model to predict label and explain the feature importance",
        dataset_id=None, df=df, trained_model_ids=[],
    )
    names = [c.name for c in calls]
    assert "train_supervised_model" in names
    explain_calls = [c for c in calls if c.name == "explain_model"]
    assert len(explain_calls) > 0
    assert explain_calls[0].arguments.get("model_id") == LATEST_TRAINED_MODEL_SENTINEL


def test_planner_routes_forecast_keyword():
    import numpy as np
    planner = Planner()
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sales": np.arange(80, dtype=float) * 0.5 + rng.normal(0, 0.5, 80),
    })
    calls = planner._rule_plan(
        "Forecast sales for the next 14 days",
        dataset_id=None, df=df, trained_model_ids=["some-model-id"],
    )
    names = [c.name for c in calls]
    assert "forecast_with_model" in names
    fc = next(c for c in calls if c.name == "forecast_with_model")
    assert fc.arguments.get("horizon") == 14


def test_planner_routes_explain_prediction_keyword():
    import numpy as np
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    calls = planner._rule_plan(
        "Why did the model predict churn for row 7?",
        dataset_id=None, df=df, trained_model_ids=["some-model-id"],
    )
    names = [c.name for c in calls]
    assert "shap_explain_prediction" in names
    ec = next(c for c in calls if c.name == "shap_explain_prediction")
    assert ec.arguments.get("row_idx") == 7


def test_planner_routes_generic_shap_to_global_explain():
    import numpy as np
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    calls = planner._rule_plan(
        "Give a shap explanation",
        dataset_id=None, df=df, trained_model_ids=["some-model-id"],
    )
    assert [c.name for c in calls] == ["explain_model"]
    assert calls[0].arguments.get("model_id") == "some-model-id"


def test_planner_keeps_shap_followup_rule_based_when_llm_enabled(monkeypatch, tmp_path):
    import numpy as np

    class FailingLLM:
        enabled = True
        def plan(self, *args, **kwargs):
            raise AssertionError("LLM planner should not handle deterministic SHAP follow-up")

    manager = DatasetManager(base_dir=str(tmp_path))
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    meta = manager.register_df(df, "dataset.csv")

    planner = Planner()
    planner.dm = manager
    planner.llm = FailingLLM()
    monkeypatch.setenv("ENABLE_RAG", "0")
    monkeypatch.setattr(planner, "_load_dataset_sample", lambda dataset_id: df)

    calls, _citations, planning_source, _llm_error, _llm_notes = planner.plan(
        "Give a shap explanation",
        dataset_id=meta.dataset_id,
        trained_model_ids=["some-model-id"],
    )
    assert planning_source == "rules"
    assert [c.name for c in calls] == ["explain_model"]


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


def test_planner_routes_pdp_keyword():
    import numpy as np
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    calls = planner._rule_plan(
        "Show me partial dependence plots for the model",
        dataset_id=None, df=df, trained_model_ids=["abc-model-id"],
    )
    names = [c.name for c in calls]
    assert "compute_pdp" in names
    pdp = next(c for c in calls if c.name == "compute_pdp")
    assert pdp.arguments.get("model_id") == "abc-model-id"


def test_planner_routes_pdp_no_trained_models_skips():
    import numpy as np
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["y"] = df["f1"] * 2 + rng.normal(0, 0.3, n)
    calls = planner._rule_plan(
        "Show me the feature effects via pdp",
        dataset_id=None, df=df, trained_model_ids=[],
    )
    names = [c.name for c in calls]
    assert "compute_pdp" not in names


def test_planner_routes_pdp_with_explicit_uuid():
    import numpy as np
    planner = Planner()
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["y"] = df["f1"] + rng.normal(0, 0.2, n)
    explicit_id = "12345678-1234-1234-1234-123456789abc"
    calls = planner._rule_plan(
        f"Compute PDP for model {explicit_id}",
        dataset_id=None, df=df, trained_model_ids=["other-model-id"],
    )
    names = [c.name for c in calls]
    assert "compute_pdp" in names
    pdp = next(c for c in calls if c.name == "compute_pdp")
    assert pdp.arguments.get("model_id") == explicit_id
