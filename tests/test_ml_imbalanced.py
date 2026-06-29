"""Tests for SMOTE-based imbalanced-class handling in supervised training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model, _SMOTE_AVAILABLE


def _balanced_clf_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    return df


def _imbalanced_clf_df(n: int = 200) -> pd.DataFrame:
    """10:1 imbalance ratio (180 majority, 20 minority)."""
    rng = np.random.default_rng(7)
    majority = pd.DataFrame({"a": rng.normal(0, 1, 180), "b": rng.normal(0, 1, 180), "label": 0})
    minority = pd.DataFrame({"a": rng.normal(5, 1, 20), "b": rng.normal(5, 1, 20), "label": 1})
    return pd.concat([majority, minority], ignore_index=True)


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


def test_smote_available():
    assert _SMOTE_AVAILABLE, "imbalanced-learn must be installed"


def test_smote_triggered_for_imbalanced_gradient_boosting(mm):
    if not _SMOTE_AVAILABLE:
        pytest.skip("imbalanced-learn not installed")
    df = _imbalanced_clf_df()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result, result.get("error")
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "smote" in notes.lower(), f"Expected SMOTE note, got: {notes}"


def test_smote_note_shows_ratio(mm):
    if not _SMOTE_AVAILABLE:
        pytest.skip()
    df = _imbalanced_clf_df()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "ratio" in notes.lower() or "imbalance" in notes.lower()


def test_smote_not_triggered_for_balanced_data(mm):
    df = _balanced_clf_df()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "smote" not in notes.lower()


def test_smote_not_triggered_for_xgboost(mm):
    """XGBoost uses scale_pos_weight — SMOTE only fires when ratio > 10."""
    if not _SMOTE_AVAILABLE:
        pytest.skip()
    df = _imbalanced_clf_df()  # 180/20 = 9.0, NOT > 10
    result = train_supervised_model(
        df, target_col="label",
        model_type="xgboost_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "smote" not in notes.lower()
