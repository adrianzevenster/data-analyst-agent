"""Tests for Tier-2 features: multi-step agentic loop, corpus index stats, anomaly explanation."""
from __future__ import annotations

import pandas as pd
import pytest

from app.analytics.anomalies import explain_anomaly


# ── Anomaly explanation ────────────────────────────────────────────────────────

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 28, 27, 100, 29],    # row 4 is extreme
        "salary": [50000, 60000, 55000, 52000, 1000000, 58000],
        "score": [0.5, 0.6, 0.55, 0.52, 0.51, 0.58],
    })


def test_explain_anomaly_returns_attributions():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=["age", "salary", "score"], row_idx=4)
    assert "error" not in result
    assert result["row_idx"] == 4
    assert "top_attributions" in result
    assert len(result["top_attributions"]) > 0
    assert "engineering_readout" in result


def test_explain_anomaly_top_feature_is_most_extreme():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=["age", "salary", "score"], row_idx=4)
    # Row 4 has extreme age (100) and salary (1M); first attribution should have highest extremeness
    attrs = result["top_attributions"]
    for i in range(len(attrs) - 1):
        assert attrs[i]["extremeness_pct"] >= attrs[i + 1]["extremeness_pct"]


def test_explain_anomaly_out_of_range():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=["age"], row_idx=999)
    assert "error" in result


def test_explain_anomaly_normal_row():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=["age", "salary", "score"], row_idx=0)
    assert "error" not in result
    # Row 0 (age=25) is low but not extreme — percentile should be low
    age_attr = next((a for a in result["top_attributions"] if a["feature"] == "age"), None)
    assert age_attr is not None
    assert age_attr["direction"] == "low"


def test_explain_anomaly_empty_cols():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=[], row_idx=0)
    assert "error" in result


def test_explain_anomaly_top_k_respected():
    df = _sample_df()
    result = explain_anomaly(df, numeric_cols=["age", "salary", "score"], row_idx=4, top_k=1)
    assert len(result["top_attributions"]) == 1


# ── Corpus index stats ─────────────────────────────────────────────────────────

def test_corpus_index_stats_endpoint(tmp_path, monkeypatch):
    """Smoke test: endpoint returns valid structure even with empty index."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.core.config as cfg_module

    # Patch the underlying data_dir which drives index_dir
    monkeypatch.setattr(cfg_module.settings, "data_dir", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/corpus/index-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_chunks" in data
    assert "unique_sources" in data
    assert "sources" in data
    assert "chunk_samples" in data
    assert data["total_chunks"] == 0


# ── Agentic continuation planner wiring ───────────────────────────────────────

def test_llm_plan_accepts_prior_step_results(monkeypatch):
    """plan() should include prior_step_results in prompt_payload when provided."""
    import json
    from app.agent.llm import LLMReasoner
    import app.core.config as cfg_module

    reasoner = LLMReasoner()
    captured = {}

    def fake_chat(messages, *, temperature=None, operation="chat", response_format=None):
        captured["payload"] = json.loads(messages[-1]["content"])
        return json.dumps({"tool_calls": []})

    monkeypatch.setattr(reasoner, "_chat", fake_chat)
    # enabled is a property that reads from settings — patch at the settings level
    monkeypatch.setattr(cfg_module.settings, "llm_enabled", True)
    monkeypatch.setattr(cfg_module.settings, "llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(cfg_module.settings, "llm_model", "test-model")

    prior = [{"tool": "anomaly_scan", "ok": True, "summary": "12 anomalies found"}]
    reasoner.plan(
        "Find outliers",
        dataset_id=None,
        df=None,
        citations=[],
        prior_step_results=prior,
    )

    assert "prior_step_results" in captured["payload"]
    assert captured["payload"]["prior_step_results"] == prior
    assert "already_executed_tools" in captured["payload"]
    assert "anomaly_scan" in captured["payload"]["already_executed_tools"]


def test_llm_plan_no_prior_step_results_omits_field(monkeypatch):
    """When prior_step_results is None, the field must NOT appear in the payload."""
    import json
    from app.agent.llm import LLMReasoner
    import app.core.config as cfg_module

    reasoner = LLMReasoner()
    captured = {}

    def fake_chat(messages, *, temperature=None, operation="chat", response_format=None):
        captured["payload"] = json.loads(messages[-1]["content"])
        return json.dumps({"tool_calls": []})

    monkeypatch.setattr(reasoner, "_chat", fake_chat)
    monkeypatch.setattr(cfg_module.settings, "llm_enabled", True)
    monkeypatch.setattr(cfg_module.settings, "llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(cfg_module.settings, "llm_model", "test-model")

    reasoner.plan("Find outliers", dataset_id=None, df=None, citations=[])

    assert "prior_step_results" not in captured["payload"]
    assert "already_executed_tools" not in captured["payload"]


# ── Tier-1 scoring latency ────────────────────────────────────────────────────

def test_scoring_latency_tracker_records_and_snapshots():
    from app.agent.latency_metrics import ScoringLatencyTracker

    tracker = ScoringLatencyTracker(window=10)
    tracker.record("model-a", 42.0)
    tracker.record("model-a", 58.0)
    tracker.record("model-b", 10.0)

    snap = tracker.snapshot()
    assert snap["n_models"] == 2
    assert "model-a" in snap["by_model"]
    assert snap["by_model"]["model-a"]["n"] == 2
    assert snap["by_model"]["model-a"]["avg_ms"] == 50.0
    assert "model-b" in snap["by_model"]


def test_scoring_latency_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/health/scoring-latency")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_models" in data
    assert "by_model" in data


# ── Tier-1 model champion ────────────────────────────────────────────────────

def test_model_manager_promote_and_get_champion(tmp_path):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from app.analytics.ml_train.model_store import ModelManager

    manager = ModelManager(base_dir=str(tmp_path))
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    # Fit a dummy pipeline
    X = np.array([[1.0], [2.0]])
    pipe.fit(X, [1.0, 2.0])

    m1 = manager.save_model(
        pipe, task_type="regression", model_type="ridge",
        target_col="y", feature_cols=["x"], dataset_id="ds1"
    )
    m2 = manager.save_model(
        pipe, task_type="regression", model_type="ridge",
        target_col="y", feature_cols=["x"], dataset_id="ds1"
    )

    assert manager.get_champion("ds1", "y") is None  # neither promoted yet

    manager.promote(m1.model_id)
    champ = manager.get_champion("ds1", "y")
    assert champ is not None
    assert champ.model_id == m1.model_id

    manager.promote(m2.model_id)
    champ2 = manager.get_champion("ds1", "y")
    assert champ2.model_id == m2.model_id
    # old champion should be demoted
    old = manager.get_meta(m1.model_id)
    assert not old.is_champion
