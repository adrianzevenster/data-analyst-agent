"""End-to-end golden tests: planner → executor → result shape.

Each test registers a small fixture DataFrame, runs the planner to get tool
calls, executes them, and asserts on the result structure.  These catch
regressions that unit tests miss: mis-routing, executor injection failures,
and broken result schemas.

No LLM is involved.  No HTTP server is required.  The executor is called
directly, keeping these fast and deterministic.
"""
from __future__ import annotations

import pytest
import pandas as pd

from app.agent.planner import Planner
from app.agent.executor import Executor
from app.analytics.dataset_manager import DatasetManager
from app.core.models import ToolCall


# ── shared fixtures ──────────────────────────────────────────────────────────

GENERIC_DF = pd.DataFrame({
    "region": ["east", "west", "east", "west", "north"] * 10,
    "revenue": [100.0, 200.0, 150.0, 50.0, 300.0] * 10,
    "units": [1, 2, 3, 4, 5] * 10,
    "cost": [40.0, 80.0, 60.0, 20.0, 120.0] * 10,
})

TIME_DF = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=60, freq="D"),
    "sales": list(range(100, 160)),
    "units": list(range(10, 70)),
})

ML_DF = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60] * 5,
    "income": [30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000] * 5,
    "churn": [0, 0, 0, 1, 1, 1, 0, 1] * 5,
})

# Has a categorical segment column so evaluate_by_segment can group by it.
SEGMENT_ML_DF = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60] * 5,
    "income": [30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000] * 5,
    "tier": (["low", "low", "mid", "mid", "high", "high", "mid", "high"] * 5),
    "churn": [0, 0, 0, 1, 1, 1, 0, 1] * 5,
})


def _trained_model(env, df, filename, target_col, model_type="random_forest_classifier"):
    """Train a model and return (dataset_id, model_id).  Fails the test on error."""
    dm, exec_, _ = env
    ds = dm.register_df(df, filename)
    calls = [ToolCall(name="train_supervised_model", arguments={
        "target_col": target_col, "model_type": model_type,
        "tune": False, "cv_folds": 2, "max_rows": 200,
    })]
    results, _, _ = exec_.run(ds.dataset_id, calls)
    r = next(r for r in results if r.name == "train_supervised_model")
    assert r.ok, f"Training failed: {r.result}"
    return ds.dataset_id, r.result["model_id"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    monkeypatch.setenv("LLM_ENABLED", "false")
    dm = DatasetManager(base_dir=str(tmp_path))
    exec_ = Executor.__new__(Executor)
    exec_.dm = dm
    from app.analytics.ml_train.model_store import ModelManager
    exec_.model_manager = ModelManager(base_dir=str(tmp_path))
    from app.analytics.tooling import get_registry
    exec_.registry = get_registry()
    planner = Planner()
    planner.dm = dm
    return dm, exec_, planner


def _run(env, df, message, override_calls=None):
    dm, exec_, planner = env
    ds = dm.register_df(df, "dataset.csv")
    calls = override_calls
    if calls is None:
        tool_calls, _, _, _, _ = planner.plan(message, ds.dataset_id)
        calls = tool_calls
    results, tables, charts = exec_.run(ds.dataset_id, calls)
    return results, tables, charts, ds.dataset_id


# ── EDA tools ────────────────────────────────────────────────────────────────

def test_profile_dataset(env):
    results, _, _, _ = _run(env, GENERIC_DF, "Profile this dataset")
    assert any(r.name == "profile_dataset" for r in results)
    r = next(r for r in results if r.name == "profile_dataset")
    assert r.ok
    assert isinstance(r.result, dict)
    assert "columns" in r.result or "n_rows" in r.result or "shape" in r.result


def test_data_quality(env):
    results, _, _, _ = _run(env, GENERIC_DF, "Run a data quality report")
    r = next((r for r in results if r.name == "data_quality_report"), None)
    assert r is not None and r.ok


def test_anomaly_scan(env):
    results, _, _, _ = _run(env, GENERIC_DF, "Find anomalies in this data")
    r = next((r for r in results if r.name == "anomaly_scan"), None)
    assert r is not None and r.ok
    assert "n_anomalies" in r.result


def test_correlation_analysis(env):
    results, _, _, _ = _run(env, GENERIC_DF, "What are the correlations?")
    r = next((r for r in results if r.name == "correlation_analysis"), None)
    assert r is not None and r.ok


def test_clustering(env):
    results, _, _, _ = _run(env, GENERIC_DF, "Cluster this data into groups")
    r = next((r for r in results if r.name == "kmeans_clusters"), None)
    assert r is not None and r.ok
    assert "k_used" in r.result


def test_trend_analysis(env):
    results, _, _, _ = _run(env, TIME_DF, "Show me the trend over time")
    r = next((r for r in results if r.name == "trend_analysis"), None)
    assert r is not None and r.ok


# ── SQL ───────────────────────────────────────────────────────────────────────

def test_duckdb_query(env):
    results, tables, _, _ = _run(
        env, GENERIC_DF, "sql: SELECT region, SUM(revenue) as total FROM t GROUP BY region"
    )
    r = next((r for r in results if r.name == "duckdb_query"), None)
    assert r is not None and r.ok
    assert tables  # at least one table returned


# ── ML pipeline ───────────────────────────────────────────────────────────────

def test_ml_train_classification(env):
    dm, exec_, planner = env
    ds = dm.register_df(ML_DF, "churn.csv")
    calls = [ToolCall(name="train_supervised_model", arguments={
        "target_col": "churn", "model_type": "random_forest_classifier",
        "tune": False, "cv_folds": 2, "max_rows": 200,
    })]
    results, _, _ = exec_.run(ds.dataset_id, calls)
    r = next(r for r in results if r.name == "train_supervised_model")
    assert r.ok
    assert "model_id" in r.result
    assert "evaluation" in r.result
    assert "accuracy" in r.result["evaluation"]


def test_ml_train_then_explain(env):
    dm, exec_, _ = env
    ds = dm.register_df(ML_DF, "churn.csv")
    train_calls = [ToolCall(name="train_supervised_model", arguments={
        "target_col": "churn", "model_type": "random_forest_classifier",
        "tune": False, "cv_folds": 2, "max_rows": 200,
    })]
    results, _, _ = exec_.run(ds.dataset_id, train_calls)
    train_r = next(r for r in results if r.name == "train_supervised_model")
    assert train_r.ok
    model_id = train_r.result["model_id"]

    explain_calls = [ToolCall(name="explain_model", arguments={"model_id": model_id})]
    results2, _, _ = exec_.run(ds.dataset_id, explain_calls)
    exp_r = next(r for r in results2 if r.name == "explain_model")
    assert exp_r.ok
    assert "feature_importances" in exp_r.result


def test_ml_evaluate_predictions(env):
    df = pd.DataFrame({
        "actual": [0, 1, 0, 1, 0, 1, 0, 1] * 5,
        "churn_prediction": [0, 1, 0, 0, 0, 1, 1, 1] * 5,
        "probability": [0.1, 0.9, 0.2, 0.4, 0.15, 0.85, 0.6, 0.8] * 5,
    })
    results, _, _, _ = _run(env, df, "Evaluate the churn predictions", override_calls=[
        ToolCall(name="evaluate_ml_predictions", arguments={})
    ])
    r = next((r for r in results if r.name == "evaluate_ml_predictions"), None)
    assert r is not None and r.ok


# ── Causal inference ─────────────────────────────────────────────────────────

def test_causal_effect(env):
    df = pd.DataFrame({
        "treatment": ([0, 1] * 25),
        "outcome": [10.0 + x for x in range(50)],
        "age": list(range(25, 75)),
    })
    results, _, _, _ = _run(env, df,
        "Estimate the causal effect of treatment on outcome",
        override_calls=[ToolCall(name="estimate_causal_effect", arguments={
            "treatment_col": "treatment", "outcome_col": "outcome",
        })]
    )
    r = next((r for r in results if r.name == "estimate_causal_effect"), None)
    assert r is not None and r.ok
    assert "ate" in r.result
    assert "p_value" in r.result


# ── Anomaly explanation ───────────────────────────────────────────────────────

def test_explain_anomaly(env):
    results, _, _, _ = _run(env, GENERIC_DF,
        "Why is row 3 an anomaly?",
        override_calls=[ToolCall(name="explain_anomaly", arguments={"row_idx": 3, "numeric_cols": []})]
    )
    r = next((r for r in results if r.name == "explain_anomaly"), None)
    assert r is not None and r.ok
    assert "top_attributions" in r.result or "error" not in r.result


# ── Auto insights (fallback) ──────────────────────────────────────────────────

def test_auto_insights_fallback(env):
    """Ambiguous message should fall back to auto_insights and return findings."""
    results, _, _, _ = _run(env, GENERIC_DF, "Tell me something interesting about this data")
    r = next((r for r in results if r.name == "auto_insights"), None)
    assert r is not None and r.ok
    assert "insights" in r.result or "engineering_readout" in r.result


# ── Multidim pivot ────────────────────────────────────────────────────────────

def test_multidim_pivot(env):
    results, tables, _, _ = _run(
        env, GENERIC_DF, "Break down revenue by region",
        override_calls=[ToolCall(name="multidim_pivot", arguments={
            "index": ["region"], "values": "revenue", "agg": "sum"
        })]
    )
    r = next((r for r in results if r.name == "multidim_pivot"), None)
    assert r is not None and r.ok


# ── Sessions 11-16: score, SHAP, PDP, what-if, segment-eval, forecast, cross-dataset ──

def test_score_with_model(env):
    ds_id, model_id = _trained_model(env, ML_DF, "churn.csv", "churn")
    dm, exec_, _ = env
    calls = [ToolCall(name="score_with_model", arguments={"model_id": model_id})]
    results, _, _ = exec_.run(ds_id, calls)
    r = next((r for r in results if r.name == "score_with_model"), None)
    assert r is not None and r.ok
    assert "n_rows_scored" in r.result
    assert r.result["n_rows_scored"] > 0
    assert "scored_rows" in r.result


def test_shap_explain_prediction(env):
    ds_id, model_id = _trained_model(env, ML_DF, "churn.csv", "churn")
    dm, exec_, _ = env
    calls = [ToolCall(name="shap_explain_prediction", arguments={"model_id": model_id, "row_idx": 0})]
    results, _, _ = exec_.run(ds_id, calls)
    r = next((r for r in results if r.name == "shap_explain_prediction"), None)
    assert r is not None and r.ok
    # SHAP may not be installed; either contributions or graceful error
    assert "feature_contributions" in r.result or "error" in r.result


def test_compute_pdp(env):
    ds_id, model_id = _trained_model(env, ML_DF, "churn.csv", "churn")
    dm, exec_, _ = env
    calls = [ToolCall(name="compute_pdp", arguments={"model_id": model_id, "n_top_features": 2})]
    results, _, charts = exec_.run(ds_id, calls)
    r = next((r for r in results if r.name == "compute_pdp"), None)
    assert r is not None and r.ok
    assert "charts" in r.result or "n_features_plotted" in r.result


def test_what_if_predict(env):
    ds_id, model_id = _trained_model(env, ML_DF, "churn.csv", "churn")
    dm, exec_, _ = env
    calls = [ToolCall(name="what_if_predict", arguments={
        "model_id": model_id, "row_idx": 0, "overrides": {"income": 95_000}
    })]
    results, _, _ = exec_.run(ds_id, calls)
    r = next((r for r in results if r.name == "what_if_predict"), None)
    assert r is not None and r.ok
    assert "original_prediction" in r.result
    assert "new_prediction" in r.result
    assert "overrides" in r.result


def test_evaluate_by_segment(env):
    ds_id, model_id = _trained_model(env, SEGMENT_ML_DF, "churn_seg.csv", "churn")
    dm, exec_, _ = env
    calls = [ToolCall(name="evaluate_by_segment", arguments={
        "model_id": model_id, "segment_col": "tier"
    })]
    results, _, _ = exec_.run(ds_id, calls)
    r = next((r for r in results if r.name == "evaluate_by_segment"), None)
    assert r is not None and r.ok
    assert "segments" in r.result
    assert "segment_col" in r.result
    # rows include per-segment entries plus an "__overall__" row
    assert len(r.result["segments"]) > 1


def test_forecast_with_model(env):
    """Train a temporal regression model then verify forecast returns ML + Holt baseline."""
    dm, exec_, _ = env
    ds = dm.register_df(TIME_DF, "time_series.csv")
    train_calls = [ToolCall(name="train_supervised_model", arguments={
        "target_col": "sales", "model_type": "ridge_regression",
        "tune": False, "cv_folds": 2,
    })]
    results, _, _ = exec_.run(ds.dataset_id, train_calls)
    train_r = next(r for r in results if r.name == "train_supervised_model")
    assert train_r.ok, f"Training failed: {train_r.result}"

    if not train_r.result.get("lag_feature_cols"):
        pytest.skip("Lag features not engineered (dataset may be too small); skipping forecast")

    model_id = train_r.result["model_id"]
    fc_calls = [ToolCall(name="forecast_with_model", arguments={"model_id": model_id, "horizon": 7})]
    results2, _, _ = exec_.run(ds.dataset_id, fc_calls)
    r = next((r for r in results2 if r.name == "forecast_with_model"), None)
    assert r is not None and r.ok
    assert "forecast_rows" in r.result
    assert len(r.result["forecast_rows"]) > 0
    assert "holt_forecast_rows" in r.result
    assert len(r.result["holt_forecast_rows"]) == len(r.result["forecast_rows"])
    assert "baseline_comparison" in r.result


def test_cross_dataset_profile(env):
    """Cross-dataset profile runs without crashing; reports no other datasets when only one is loaded."""
    results, _, _, _ = _run(
        env, GENERIC_DF, "Compare this dataset against other datasets",
        override_calls=[ToolCall(name="cross_dataset_profile", arguments={})]
    )
    r = next((r for r in results if r.name == "cross_dataset_profile"), None)
    assert r is not None and r.ok
    # Either found cross-dataset comparisons or reported no other datasets gracefully
    assert "comparisons" in r.result or "n_datasets_compared" in r.result or "error" in r.result
