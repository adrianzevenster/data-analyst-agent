from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_eval.classification import evaluate_classification
from app.analytics.ml_eval.regression import evaluate_regression_or_forecast
from app.analytics.ml_eval.columns import infer_prediction_columns
from app.analytics.ml_eval.report import evaluate_ml_predictions


def test_classification_drops_nan_rows_before_scoring():
    df = pd.DataFrame(
        {
            "actual": [1, 0, None, 1],
            "prediction": [1, 1, 1, None],
        }
    )

    result = evaluate_classification(df, actual_col="actual", prediction_col="prediction")

    # Only the two fully-populated rows are usable.
    assert result["n_rows_evaluated"] == 2
    assert result["accuracy"] == 0.5


def test_classification_all_nan_returns_error_not_crash():
    df = pd.DataFrame({"actual": [None, None], "prediction": [None, None]})

    result = evaluate_classification(df, actual_col="actual", prediction_col="prediction")

    assert "error" in result
    assert result["task_type"] == "classification"


def test_classification_single_class_labels_skips_roc_auc():
    df = pd.DataFrame(
        {
            "actual": [1, 1, 1, 1],
            "prediction": [1, 1, 1, 0],
            "probability": [0.9, 0.8, 0.7, 0.6],
        }
    )

    result = evaluate_classification(
        df, actual_col="actual", prediction_col="prediction", probability_col="probability"
    )

    # roc_auc/average_precision require exactly 2 distinct true labels.
    assert "roc_auc" not in result
    assert "average_precision" not in result
    assert result["accuracy"] == 0.75


def test_classification_non_numeric_probability_column_is_ignored_not_fatal():
    df = pd.DataFrame(
        {
            "actual": [1, 0, 1, 0],
            "prediction": [1, 1, 1, 0],
            "probability": ["high", "low", "medium", "low"],
        }
    )

    result = evaluate_classification(
        df, actual_col="actual", prediction_col="prediction", probability_col="probability"
    )

    assert "probability" not in result
    assert result["accuracy"] == 0.75


def test_classification_string_labels_compute_without_numeric_coercion():
    df = pd.DataFrame(
        {
            "actual": ["yes", "no", "yes", "no"],
            "prediction": ["yes", "yes", "yes", "no"],
        }
    )

    result = evaluate_classification(df, actual_col="actual", prediction_col="prediction")

    assert result["labels"] == ["no", "yes"]
    assert result["accuracy"] == 0.75
    # Non-numeric labels mean roc_auc can't be computed even with 2 classes.
    assert "roc_auc" not in result


def test_evaluate_ml_predictions_unrecoverable_schema_returns_actionable_error():
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["x", "y", "z"]})

    result = evaluate_ml_predictions(df)

    assert result["task_type"] == "unknown"
    assert "error" in result
    assert "Could not infer ML prediction structure" in result["error"]


def test_evaluate_ml_predictions_regression_with_nans_drops_incomplete_rows():
    df = pd.DataFrame({"actual": [100.0, None, 300.0], "prediction": [90.0, 210.0, None]})

    result = evaluate_ml_predictions(df, task_hint="regression")

    assert result["task_type"] == "regression"
    assert "error" not in result


def test_infer_prediction_columns_returns_none_when_nothing_matches():
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})

    inferred = infer_prediction_columns(df)

    assert inferred.actual_col is None
    assert inferred.prediction_col is None
    assert inferred.probability_col is None
    assert inferred.id_col is None


def test_infer_prediction_columns_disambiguates_churn_probability_without_labels():
    # churn_probability + churn_prediction but no actual/label column: should be
    # treated as scored predictions, not misread "churn" as the actual column.
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "churn_probability": [0.9, 0.2, 0.6],
            "churn_prediction": [1, 0, 1],
        }
    )

    inferred = infer_prediction_columns(df)

    assert inferred.probability_col == "churn_probability"
    assert inferred.prediction_col == "churn_prediction"


# ---------------------------------------------------------------------------
# Eval chart specs — classification
# ---------------------------------------------------------------------------

def _binary_clf_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    actual = rng.integers(0, 2, n)
    prob = np.clip(actual * 0.6 + rng.normal(0, 0.2, n), 0, 1)
    pred = (prob > 0.5).astype(int)
    return pd.DataFrame({"actual": actual, "prediction": pred, "probability": prob})


def test_roc_curve_present_for_binary_with_probability():
    df = _binary_clf_df()
    result = evaluate_classification(df, "actual", "prediction", probability_col="probability")
    assert "roc_curve" in result
    roc = result["roc_curve"]
    assert roc["type"] == "line"
    assert roc["x"] == "fpr"
    assert roc["y"] == "tpr"
    assert len(roc["data"]) > 2
    # All points should have fpr in [0,1] and tpr in [0,1]
    for pt in roc["data"]:
        assert 0 <= pt["fpr"] <= 1
        assert 0 <= pt["tpr"] <= 1


def test_pr_curve_present_for_binary_with_probability():
    df = _binary_clf_df()
    result = evaluate_classification(df, "actual", "prediction", probability_col="probability")
    assert "pr_curve" in result
    pr = result["pr_curve"]
    assert pr["type"] == "line"
    assert pr["x"] == "recall"
    assert pr["y"] == "precision"
    assert len(pr["data"]) > 2
    for pt in pr["data"]:
        assert 0 <= pt["recall"] <= 1
        assert 0 <= pt["precision"] <= 1


def test_curves_absent_when_no_probability_col():
    df = _binary_clf_df()
    result = evaluate_classification(df, "actual", "prediction")
    assert "roc_curve" not in result
    assert "pr_curve" not in result


def test_curves_absent_for_multiclass():
    df = pd.DataFrame({"actual": [0, 1, 2, 0, 1, 2], "prediction": [0, 2, 2, 0, 1, 1]})
    result = evaluate_classification(df, "actual", "prediction")
    assert "roc_curve" not in result
    assert "pr_curve" not in result


def test_confusion_matrix_always_present():
    df = _binary_clf_df()
    result = evaluate_classification(df, "actual", "prediction")
    assert "confusion_matrix" in result
    cm = result["confusion_matrix"]
    assert "labels" in cm
    assert "matrix" in cm
    mat = cm["matrix"]
    assert len(mat) == 2
    assert all(len(row) == 2 for row in mat)


# ---------------------------------------------------------------------------
# Eval chart specs — regression
# ---------------------------------------------------------------------------

def _reg_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    actual = rng.normal(10, 3, n)
    prediction = actual + rng.normal(0, 0.5, n)
    return pd.DataFrame({"actual": actual, "prediction": prediction})


def test_actual_vs_predicted_present_for_regression():
    df = _reg_df()
    result = evaluate_regression_or_forecast(df, "actual", "prediction")
    assert "actual_vs_predicted" in result
    avp = result["actual_vs_predicted"]
    assert avp["type"] == "scatter"
    assert avp["x"] == "actual"
    assert avp["y"] == "prediction"
    assert len(avp["data"]) > 0
    for pt in avp["data"]:
        assert "actual" in pt
        assert "prediction" in pt


def test_actual_vs_predicted_capped_at_200_points():
    df = _reg_df(n=300)
    result = evaluate_regression_or_forecast(df, "actual", "prediction")
    avp = result["actual_vs_predicted"]
    assert len(avp["data"]) <= 200


def test_residuals_hist_present_for_regression():
    df = _reg_df()
    result = evaluate_regression_or_forecast(df, "actual", "prediction")
    assert "residuals_hist" in result
    rh = result["residuals_hist"]
    assert rh["type"] == "histogram"
    assert len(rh["data"]) >= 10
    for bucket in rh["data"]:
        assert "bin_label" in bucket
        assert "count" in bucket
        assert bucket["count"] >= 0


def test_regression_chart_specs_absent_when_all_nan():
    df = pd.DataFrame({"actual": [None, None], "prediction": [None, None]})
    result = evaluate_regression_or_forecast(df, "actual", "prediction")
    assert "error" in result
    assert "actual_vs_predicted" not in result
    assert "residuals_hist" not in result
