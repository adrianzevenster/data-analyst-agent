"""Tests for partial dependence plots (compute_pdp)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.pdp import compute_pdp


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(0, 0.3, n)
    return df


def _clf_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


def test_pdp_returns_charts(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    assert "error" not in res
    pdp_result = compute_pdp(df, res["model_id"], model_manager=mm, n_top_features=2)
    assert "error" not in pdp_result, pdp_result.get("error")
    assert pdp_result["n_features_plotted"] >= 1
    assert len(pdp_result["charts"]) >= 1


def test_pdp_chart_structure(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    pdp_result = compute_pdp(df, res["model_id"], model_manager=mm, n_top_features=2)
    for chart in pdp_result["charts"]:
        assert chart["type"] == "line"
        assert chart["x"] == "value"
        assert chart["y"] == "effect"
        assert len(chart["data"]) >= 2
        for pt in chart["data"]:
            assert "value" in pt
            assert "effect" in pt
            assert isinstance(pt["effect"], float)


def test_pdp_for_classifier(mm):
    df = _clf_df(n=120)
    res = train_supervised_model(
        df, target_col="label", model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in res
    pdp_result = compute_pdp(df, res["model_id"], model_manager=mm, n_top_features=2)
    assert pdp_result["n_features_plotted"] >= 1
    for chart in pdp_result["charts"]:
        for pt in chart["data"]:
            assert 0.0 <= pt["effect"] <= 1.0


def test_pdp_missing_model_returns_error(mm):
    df = _reg_df()
    result = compute_pdp(df, "nonexistent-model-id", model_manager=mm)
    assert "error" in result


def test_pdp_missing_features_returns_error(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    result = compute_pdp(pd.DataFrame({"z": [1, 2, 3]}), res["model_id"], model_manager=mm)
    assert "error" in result
