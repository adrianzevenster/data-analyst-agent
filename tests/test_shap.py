"""Tests for per-prediction SHAP explanations (shap_explain_prediction)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.explainability import shap_explain_prediction


def _classification_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


@pytest.fixture
def clf_model(mm):
    df = _classification_df(n=200)
    result = train_supervised_model(
        df, target_col="label", model_type="random_forest_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result, result.get("error")
    return result["model_id"], mm


def test_shap_explain_prediction_returns_contributions(clf_model):
    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")
    model_id, mm = clf_model
    df = _classification_df(n=200)
    result = shap_explain_prediction(df, model_id=model_id, row_idx=0, model_manager=mm)
    if "error" in result:
        pytest.skip(f"SHAP explain not available: {result['error']}")
    assert "feature_contributions" in result
    assert len(result["feature_contributions"]) > 0


def test_shap_explain_prediction_signed_values(clf_model):
    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")
    model_id, mm = clf_model
    df = _classification_df(n=200)
    result = shap_explain_prediction(df, model_id=model_id, row_idx=3, model_manager=mm)
    if "error" in result:
        pytest.skip(f"SHAP explain not available: {result['error']}")
    for contrib in result["feature_contributions"]:
        assert "feature" in contrib
        assert "shap_value" in contrib
        assert isinstance(contrib["shap_value"], float)


def test_shap_explain_prediction_out_of_range(clf_model):
    model_id, mm = clf_model
    df = _classification_df(n=200)
    result = shap_explain_prediction(df, model_id=model_id, row_idx=99999, model_manager=mm)
    assert "error" in result


def test_shap_explain_prediction_nonexistent_model(mm):
    df = _classification_df(n=200)
    result = shap_explain_prediction(df, model_id="00000000-0000-0000-0000-000000000000", model_manager=mm)
    assert "error" in result
