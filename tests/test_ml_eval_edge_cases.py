from __future__ import annotations

import pandas as pd

from app.analytics.ml_eval.classification import evaluate_classification
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
