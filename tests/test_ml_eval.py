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
