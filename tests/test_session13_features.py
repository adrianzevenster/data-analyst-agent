"""Tests for session-13 features:
- Automated feature selection (near-constant + correlation filter)
- Calibration curve in classification eval
- Partial dependence plots (PDPs)
- Per-class metrics table (via classification_report in eval)
- CSV download (client-side, no backend test needed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_eval.classification import evaluate_classification
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import select_features
from app.analytics.ml_train.training import train_supervised_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    return df


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(0, 0.3, n)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# 1. Feature selection
# ---------------------------------------------------------------------------

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
    # One of a/b should be dropped (r > 0.95); c should survive
    assert len(selected) == 2
    assert "c" in selected
    assert any("correlated" in n for n in notes)


def test_select_features_returns_all_when_no_issues():
    df = _reg_df(n=100)
    selected, notes = select_features(df, ["x1", "x2"])
    assert set(selected) == {"x1", "x2"}
    assert notes == []


def test_feature_selection_notes_appear_in_training_result(mm):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 0.0001, 100),  # near-duplicate of a → r > 0.95
        "const": [5.0] * 100,                   # constant → near-zero std
        "target": rng.normal(0, 1, 100),
    })
    result = train_supervised_model(df, target_col="target", tune=False, cv_folds=2, model_manager=mm)
    assert "error" not in result, result.get("error")
    notes = " ".join(result.get("preprocessing_notes", []))
    # At minimum the constant column should be flagged
    assert "auto-dropped" in notes.lower()


# ---------------------------------------------------------------------------
# 2. Calibration curve
# ---------------------------------------------------------------------------

def _binary_clf_eval_df(n: int = 100) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(3)
    actual = rng.integers(0, 2, n)
    prob = np.clip(actual * 0.7 + rng.normal(0, 0.15, n), 0, 1)
    pred = (prob > 0.5).astype(int)
    df = pd.DataFrame({"actual": actual, "prediction": pred, "probability": prob})
    result = evaluate_classification(df, "actual", "prediction", probability_col="probability")
    return df, result


def test_calibration_curve_present_for_binary():
    _, result = _binary_clf_eval_df()
    assert "calibration_curve" in result
    cc = result["calibration_curve"]
    assert cc["type"] == "line"
    assert cc["x"] == "mean_predicted"
    assert cc["y"] == "fraction_positive"
    assert len(cc["data"]) >= 2


def test_calibration_curve_values_in_unit_interval():
    _, result = _binary_clf_eval_df()
    for pt in result["calibration_curve"]["data"]:
        assert 0 <= pt["mean_predicted"] <= 1
        assert 0 <= pt["fraction_positive"] <= 1


def test_calibration_curve_absent_without_probability():
    df = _clf_df()
    df["pred"] = df["label"]
    result = evaluate_classification(df, "label", "pred")  # no prob col
    assert "calibration_curve" not in result


# ---------------------------------------------------------------------------
# 3. Per-class metrics (classification_report)
# ---------------------------------------------------------------------------

def test_classification_report_present():
    df = _clf_df()
    df["pred"] = df["label"]
    result = evaluate_classification(df, "label", "pred")
    assert "classification_report" in result
    report = result["classification_report"]
    class_keys = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    assert len(class_keys) >= 2


def test_classification_report_has_per_class_fields():
    df = _clf_df()
    df["pred"] = df["label"]
    result = evaluate_classification(df, "label", "pred")
    report = result["classification_report"]
    for k, v in report.items():
        if k in ("accuracy",):
            continue
        assert "precision" in v
        assert "recall" in v
        assert "f1-score" in v
        assert "support" in v


# ---------------------------------------------------------------------------
# 4. Partial dependence plots
# ---------------------------------------------------------------------------

def test_pdp_returns_charts(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    assert "error" not in res

    from app.analytics.ml_train.pdp import compute_pdp
    pdp_result = compute_pdp(df, res["model_id"], model_manager=mm, n_top_features=2)
    assert "error" not in pdp_result, pdp_result.get("error")
    assert pdp_result["n_features_plotted"] >= 1
    assert len(pdp_result["charts"]) >= 1


def test_pdp_chart_structure(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    from app.analytics.ml_train.pdp import compute_pdp
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
    from app.analytics.ml_train.pdp import compute_pdp
    pdp_result = compute_pdp(df, res["model_id"], model_manager=mm, n_top_features=2)
    assert pdp_result["n_features_plotted"] >= 1
    # For binary classifier, effect values should be in [0, 1] (predicted probability)
    for chart in pdp_result["charts"]:
        for pt in chart["data"]:
            assert 0.0 <= pt["effect"] <= 1.0


def test_pdp_missing_model_returns_error(mm):
    from app.analytics.ml_train.pdp import compute_pdp
    df = _reg_df()
    result = compute_pdp(df, "nonexistent-model-id", model_manager=mm)
    assert "error" in result


def test_pdp_missing_features_returns_error(mm):
    df = _reg_df(n=100)
    res = train_supervised_model(df, target_col="y", tune=False, cv_folds=2, model_manager=mm)
    from app.analytics.ml_train.pdp import compute_pdp
    result = compute_pdp(pd.DataFrame({"z": [1, 2, 3]}), res["model_id"], model_manager=mm)
    assert "error" in result
