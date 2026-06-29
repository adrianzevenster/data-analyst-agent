"""Tests for conformal prediction sets on classification models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.scoring import score_with_model


def _balanced_clf_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


def test_conformal_threshold_stored(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    meta = mm.get_meta(result["model_id"])
    assert meta.conformal_classification_threshold is not None
    assert 0.0 <= meta.conformal_classification_threshold <= 1.0


def test_prediction_set_in_scoring(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    scored = score_with_model(df, model_id=result["model_id"], model_manager=mm)
    assert "prediction_set_info" in scored
    ps_info = scored["prediction_set_info"]
    assert ps_info is not None
    assert ps_info["coverage_target"] == 0.90
    assert ps_info["avg_set_size"] >= 1.0
    rows = scored["scored_rows"]
    assert len(rows) > 0
    assert "prediction_set" in rows[0]


def test_prediction_set_values_are_valid(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    scored = score_with_model(df, model_id=result["model_id"], model_manager=mm)
    rows = scored["scored_rows"]
    for row in rows[:20]:
        ps = row.get("prediction_set", "")
        for cls in str(ps).split("|"):
            assert cls in ("0", "1"), f"Unexpected class in prediction_set: {cls!r}"
