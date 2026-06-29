"""Tests for automated feature selection (near-constant and correlation filter)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import select_features
from app.analytics.ml_train.training import train_supervised_model


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


def test_select_features_drops_constant_numeric():
    df = pd.DataFrame({"a": [1.0] * 50, "b": np.random.default_rng(0).normal(0, 1, 50)})
    selected, notes = select_features(df, ["a", "b"])
    assert "a" not in selected
    assert "b" in selected
    assert any("near-constant" in n for n in notes)


def test_select_features_drops_constant_categorical():
    df = pd.DataFrame({"cat": ["x"] * 50, "num": range(50)})
    selected, notes = select_features(df, ["cat", "num"])
    assert "cat" not in selected
    assert "num" in selected


def test_select_features_drops_highly_correlated():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.001, 100), "c": rng.normal(0, 1, 100)})
    selected, notes = select_features(df, ["a", "b", "c"])
    assert len(selected) == 2
    assert "c" in selected
    assert any("correlated" in n for n in notes)


def test_select_features_returns_all_when_no_issues():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(0, 1, 100), "x2": rng.normal(0, 1, 100)})
    selected, notes = select_features(df, ["x1", "x2"])
    assert set(selected) == {"x1", "x2"}
    assert notes == []


def test_feature_selection_notes_appear_in_training_result(mm):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 0.0001, 100),
        "const": [5.0] * 100,
        "target": rng.normal(0, 1, 100),
    })
    result = train_supervised_model(df, target_col="target", tune=False, cv_folds=2, model_manager=mm)
    assert "error" not in result, result.get("error")
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "auto-dropped" in notes.lower()
