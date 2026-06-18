"""
Integration tests for the ML pipeline: train → compare → explain, via the chat endpoint.

Tests use TestClient (ASGI in-process) so they catch serialization bugs
that only surface at the HTTP boundary (numpy scalars, NaN, etc.).
"""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_SALES_CSV = """region,revenue,units,cost
east,100.0,1,40.0
west,200.0,2,80.0
east,150.0,3,60.0
west,50.0,4,20.0
north,300.0,5,120.0
south,80.0,1,32.0
east,220.0,6,88.0
west,170.0,3,68.0
north,90.0,2,36.0
south,130.0,4,52.0
east,75.0,1,30.0
west,310.0,7,124.0
"""

_CHURN_CSV = ",".join(["age", "tenure", "monthly_charges", "churn"]) + "\n" + "\n".join(
    f"{30 + i},{i % 24},{30 + (i % 50)},{int(i % 3 == 0)}"
    for i in range(60)
)


@pytest.fixture(scope="module")
def sales_dataset_id(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "sales.csv"
    csv_path.write_text(_SALES_CSV)
    with open(csv_path, "rb") as f:
        resp = client.post("/uploads", files={"file": ("sales.csv", f, "text/csv")})
    assert resp.status_code == 200
    return resp.json()["dataset_id"]


@pytest.fixture(scope="module")
def churn_dataset_id(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    csv_path = tmp / "churn.csv"
    csv_path.write_text(_CHURN_CSV)
    with open(csv_path, "rb") as f:
        resp = client.post("/uploads", files={"file": ("churn.csv", f, "text/csv")})
    assert resp.status_code == 200
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Numpy serialization regression
# ---------------------------------------------------------------------------

def test_profile_response_is_json_serializable(sales_dataset_id):
    """Numpy scalars in profile output must not escape into the HTTP response."""
    resp = client.post("/chat", json={"dataset_id": sales_dataset_id, "message": "Profile this dataset"})
    assert resp.status_code == 200
    body = resp.text
    # If numpy types leaked, FastAPI raises a 500; a 200 proves serialization is clean.
    data = resp.json()
    assert isinstance(data["tool_results"], list)
    assert any(tr["ok"] for tr in data["tool_results"])


def test_correlation_response_is_json_serializable(sales_dataset_id):
    resp = client.post("/chat", json={"dataset_id": sales_dataset_id, "message": "Show correlations"})
    assert resp.status_code == 200
    data = resp.json()
    assert any(tr["ok"] for tr in data["tool_results"])


# ---------------------------------------------------------------------------
# Train → explain
# ---------------------------------------------------------------------------

def test_train_returns_feature_importance(churn_dataset_id):
    resp = client.post(
        "/chat",
        json={
            "dataset_id": churn_dataset_id,
            "message": "Train a model to predict churn",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    train_results = [tr for tr in data["tool_results"] if tr["name"] == "train_supervised_model"]
    assert train_results, "Expected train_supervised_model to run"
    result = train_results[0]
    assert result["ok"], f"Training failed: {result.get('error')}"
    assert "feature_importance" in result["result"]
    assert isinstance(result["result"]["feature_importance"], list)


def test_train_then_explain(churn_dataset_id):
    # Train a model via chat
    train_resp = client.post(
        "/chat",
        json={
            "dataset_id": churn_dataset_id,
            "message": "Train a random forest to predict churn",
        },
    )
    assert train_resp.status_code == 200
    train_body = train_resp.json()
    train_tr = next(
        (tr for tr in train_body["tool_results"] if tr["name"] == "train_supervised_model" and tr["ok"]),
        None,
    )
    assert train_tr is not None, "Training must succeed"
    model_id = train_tr["result"]["model_id"]

    # Now explain the model
    explain_resp = client.post(
        "/chat",
        json={
            "dataset_id": churn_dataset_id,
            "message": f"Explain the model {model_id}",
        },
    )
    assert explain_resp.status_code == 200
    explain_body = explain_resp.json()
    explain_tr = next(
        (tr for tr in explain_body["tool_results"] if tr["name"] == "explain_model"),
        None,
    )
    assert explain_tr is not None, "explain_model tool should have run"
    assert explain_tr["ok"], f"explain_model failed: {explain_tr.get('error')}"
    result = explain_tr["result"]
    assert "feature_importances" in result
    assert len(result["feature_importances"]) > 0
    assert result["target_col"] == "churn"


# ---------------------------------------------------------------------------
# Model comparison on retrain
# ---------------------------------------------------------------------------

def test_retrain_produces_model_comparison(churn_dataset_id):
    # First train
    r1 = client.post(
        "/chat",
        json={
            "dataset_id": churn_dataset_id,
            "message": "Train a logistic regression to predict churn",
        },
    )
    assert r1.status_code == 200
    tr1 = next(
        (tr for tr in r1.json()["tool_results"] if tr["name"] == "train_supervised_model" and tr["ok"]),
        None,
    )
    assert tr1 is not None, "First training must succeed"

    # Second train on same dataset + target
    r2 = client.post(
        "/chat",
        json={
            "dataset_id": churn_dataset_id,
            "message": "Train a random forest to predict churn",
        },
    )
    assert r2.status_code == 200
    tr2 = next(
        (tr for tr in r2.json()["tool_results"] if tr["name"] == "train_supervised_model" and tr["ok"]),
        None,
    )
    assert tr2 is not None, "Second training must succeed"

    comparison = tr2["result"].get("model_comparison")
    assert comparison is not None, "model_comparison must be present on retrain"
    assert comparison["metric"] == "accuracy"
    assert "previous_model_id" in comparison
    assert "delta" in comparison
    assert isinstance(comparison["improved"], bool)


# ---------------------------------------------------------------------------
# explain_model tool via direct call (unit-level sanity)
# ---------------------------------------------------------------------------

def test_explain_model_direct(churn_dataset_id):
    from app.analytics.ml_train.model_store import ModelManager

    # Grab the most recent model for churn
    mgr = ModelManager()
    models = [m for m in mgr.list_models() if m.target_col == "churn"]
    if not models:
        pytest.skip("No churn models in registry — run test_train_then_explain first")
    model_id = max(models, key=lambda m: m.created_at).model_id

    from app.analytics.dataset_manager import DatasetManager
    dm = DatasetManager()
    df = dm.load_df(churn_dataset_id)

    from app.analytics.ml_train.explainability import explain_model
    result = explain_model(df, model_id=model_id)

    assert "error" not in result, f"explain_model returned error: {result.get('error')}"
    assert result["target_col"] == "churn"
    assert isinstance(result["feature_importances"], list)
    assert len(result["feature_importances"]) > 0
    fi = result["feature_importances"][0]
    assert "feature" in fi and "importance_mean" in fi and "importance_std" in fi


# ---------------------------------------------------------------------------
# Judge history endpoint
# ---------------------------------------------------------------------------

def test_judge_history_endpoint_returns_list():
    resp = client.get("/health/llm-judge/history")
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body
    assert "total" in body
    assert isinstance(body["entries"], list)
    assert body["total"] == len(body["entries"])
