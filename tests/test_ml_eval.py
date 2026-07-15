from __future__ import annotations

import pandas as pd

from app.analytics.ml_eval.columns import infer_prediction_columns
from app.analytics.ml_eval.report import evaluate_ml_predictions


def test_infer_prediction_columns_for_classification_dataset():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "actual": [1, 0],
            "prediction": [1, 1],
            "probability": [0.9, 0.7],
        }
    )

    inferred = infer_prediction_columns(df)

    assert inferred.id_col == "customer_id"
    assert inferred.actual_col == "actual"
    assert inferred.prediction_col == "prediction"
    assert inferred.probability_col == "probability"


def test_evaluate_ml_predictions_classification_metrics():
    df = pd.DataFrame(
        {
            "actual": [1, 0, 1, 0],
            "prediction": [1, 1, 1, 0],
            "probability": [0.9, 0.8, 0.7, 0.1],
            "segment": ["a", "a", "b", "b"],
        }
    )

    result = evaluate_ml_predictions(df, task_hint="classification", slice_cols=["segment"])

    assert result["task_type"] == "classification"
    assert result["evaluation"]["accuracy"] == 0.75
    assert result["evaluation"]["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert result["worst_error_slices"]
    assert "Classification evaluation complete" in result["engineering_readout"]


def test_evaluate_ml_predictions_regression_metrics():
    df = pd.DataFrame({"actual": [100, 200, 300], "prediction": [90, 210, 330]})

    result = evaluate_ml_predictions(df, task_hint="regression")

    assert result["task_type"] == "regression"
    assert result["evaluation"]["mae"] == 50 / 3
    assert result["evaluation"]["wmape"] == 50 / 600
    assert "Regression/forecast evaluation complete" in result["engineering_readout"]


def test_evaluate_ml_predictions_scored_predictions_without_labels():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "churn_probability": [0.91, 0.42, 0.12],
            "churn_prediction": [1, 0, 0],
        }
    )

    result = evaluate_ml_predictions(df, task_hint="scored_predictions")

    assert result["task_type"] == "scored_predictions"
    assert result["evaluation"]["confidence_bands"]["high_confidence_0_80_plus"] == 1
    assert "Scored prediction evaluation complete" in result["engineering_readout"]


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def _binary_clf_eval_df(n: int = 100):
    import numpy as np
    from app.analytics.ml_eval.classification import evaluate_classification
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
    assert "fraction_positive" in cc.get("y_series", [cc.get("y")])
    assert len(cc["data"]) >= 2


def test_calibration_curve_values_in_unit_interval():
    _, result = _binary_clf_eval_df()
    for pt in result["calibration_curve"]["data"]:
        assert 0 <= pt["mean_predicted"] <= 1
        assert 0 <= pt["fraction_positive"] <= 1


def test_calibration_curve_absent_without_probability():
    import numpy as np
    from app.analytics.ml_eval.classification import evaluate_classification
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    df["pred"] = df["label"]
    result = evaluate_classification(df, "label", "pred")
    assert "calibration_curve" not in result


# ---------------------------------------------------------------------------
# Per-class metrics (classification_report)
# ---------------------------------------------------------------------------

def test_classification_report_present():
    import numpy as np
    from app.analytics.ml_eval.classification import evaluate_classification
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    df["pred"] = df["label"]
    result = evaluate_classification(df, "label", "pred")
    assert "classification_report" in result
    report = result["classification_report"]
    class_keys = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    assert len(class_keys) >= 2


def test_classification_report_has_per_class_fields():
    import numpy as np
    from app.analytics.ml_eval.classification import evaluate_classification
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
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
