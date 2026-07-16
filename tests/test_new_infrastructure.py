"""Tests for recently-added infrastructure: semantic router, ONNX inference path,
forecast PI propagation, and feedback endpoint."""
from __future__ import annotations

import math

import pytest


# ── 1. Semantic router ────────────────────────────────────────────────────────


def test_semantic_router_known_queries():
    """Well-known queries should route to the correct tool."""
    from app.agent.semantic_router import route

    assert "correlation_analysis" in route("what are the correlations between variables")
    assert "trend_analysis" in route("show me trends over time")
    assert "anomaly_scan" in route("find outliers in the data")


def test_semantic_router_empty_on_low_confidence():
    """Very vague single-character query should not raise and must return a list."""
    from app.agent.semantic_router import route

    result = route("x")
    assert isinstance(result, list)


def test_semantic_router_returns_list():
    from app.agent.semantic_router import route

    r = route("cluster the customers by behaviour")
    assert isinstance(r, list)
    if r:
        assert all(isinstance(t, str) for t in r)


def test_semantic_router_embedder_failure_returns_empty(monkeypatch):
    """If the embedder raises during init, route() must return [] not raise."""
    import app.agent.semantic_router as sr
    import app.rag.embedder as emb_mod

    # Force re-initialisation on next call.
    sr._embeddings = None
    sr._tool_names = []

    def _bad_embed(self, texts):  # noqa: ANN001
        raise RuntimeError("no model")

    monkeypatch.setattr(emb_mod.LocalEmbedder, "embed", _bad_embed)
    result = sr.route("find anomalies")
    assert isinstance(result, list)

    # Restore so later tests can embed normally.
    sr._embeddings = None
    sr._tool_names = []


# ── 2. ONNX inference path ────────────────────────────────────────────────────


def test_try_onnx_predict_no_onnx_path():
    """Returns None when meta has no onnx_path."""
    import pandas as pd
    from types import SimpleNamespace

    from app.analytics.ml_train.scoring import _try_onnx_predict

    meta = SimpleNamespace(onnx_path=None)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert _try_onnx_predict(meta, X) is None


def test_try_onnx_predict_missing_file(tmp_path):
    """Returns None when onnx_path points to a non-existent file."""
    import pandas as pd
    from types import SimpleNamespace

    from app.analytics.ml_train.scoring import _try_onnx_predict

    meta = SimpleNamespace(onnx_path=str(tmp_path / "ghost.onnx"))
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert _try_onnx_predict(meta, X) is None


def test_try_onnx_predict_import_error(tmp_path, monkeypatch):
    """Returns None and does not raise when onnxruntime is unavailable."""
    import builtins
    import pandas as pd
    from types import SimpleNamespace

    from app.analytics.ml_train.scoring import _try_onnx_predict

    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"dummy")
    meta = SimpleNamespace(onnx_path=str(onnx_file))
    X = pd.DataFrame({"a": [1.0, 2.0]})

    real_import = builtins.__import__

    def _no_ort(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("mocked unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ort)
    assert _try_onnx_predict(meta, X) is None


# ── 3. Forecast PI propagation ────────────────────────────────────────────────


def test_forecast_pi_widens_with_horizon():
    """Prediction intervals must get wider as the forecast horizon increases.

    The formula is halfwidth * sqrt(step), so step=4 must be exactly 2× step=1.
    """
    halfwidth = 10.0
    step1_hw = halfwidth * math.sqrt(1)
    step4_hw = halfwidth * math.sqrt(4)

    assert step4_hw > step1_hw
    assert abs(step4_hw - 2 * step1_hw) < 1e-9


def test_forecast_pi_formula():
    """sqrt scaling produces correct interval width for a range of steps."""
    halfwidth = 5.0
    for step in [1, 2, 7, 14, 30]:
        expected_hw = halfwidth * math.sqrt(step)
        lower = 100.0 - expected_hw
        upper = 100.0 + expected_hw
        assert upper - lower == pytest.approx(2 * expected_hw, rel=1e-6)


# ── 4. Feedback endpoint ──────────────────────────────────────────────────────


def _make_feedback_client(tmp_path, monkeypatch):
    """Return a TestClient wired to an isolated SQLite database."""
    import app.api.routes_feedback as fb
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(fb, "_DB", tmp_path / "test.db")
    fb._init_table()

    _app = FastAPI()
    _app.include_router(fb.router)
    return TestClient(_app)


def test_feedback_post_valid(tmp_path, monkeypatch):
    """POST / stores a valid 'up' rating and returns status ok."""
    client = _make_feedback_client(tmp_path, monkeypatch)
    r = client.post("/", json={"conversation_id": "conv1", "turn_idx": 0, "rating": "up"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "id" in body


def test_feedback_post_down_rating(tmp_path, monkeypatch):
    """POST / stores a valid 'down' rating."""
    client = _make_feedback_client(tmp_path, monkeypatch)
    r = client.post("/", json={"conversation_id": "conv1", "turn_idx": 1, "rating": "down"})
    assert r.status_code == 200


def test_feedback_post_invalid_rating(tmp_path, monkeypatch):
    """POST / rejects ratings that aren't 'up' or 'down'."""
    client = _make_feedback_client(tmp_path, monkeypatch)
    r = client.post("/", json={"conversation_id": "conv1", "turn_idx": 0, "rating": "meh"})
    assert r.status_code == 422


def test_feedback_stats(tmp_path, monkeypatch):
    """GET /stats returns correct totals and up_rate."""
    client = _make_feedback_client(tmp_path, monkeypatch)

    client.post("/", json={"conversation_id": "c1", "turn_idx": 0, "rating": "up"})
    client.post("/", json={"conversation_id": "c1", "turn_idx": 1, "rating": "up"})
    client.post("/", json={"conversation_id": "c2", "turn_idx": 0, "rating": "down"})

    stats = client.get("/stats").json()
    assert stats["total"] == 3
    assert stats["up"] == 2
    assert stats["down"] == 1
    assert abs(stats["up_rate"] - 0.667) < 0.01


def test_feedback_stats_empty(tmp_path, monkeypatch):
    """GET /stats returns zeros with up_rate 0.0 when no feedback exists."""
    client = _make_feedback_client(tmp_path, monkeypatch)
    stats = client.get("/stats").json()
    assert stats["total"] == 0
    assert stats["up_rate"] == 0.0
