from __future__ import annotations

from typing import Literal

import pandas as pd

from app.analytics.ml_eval.classification import evaluate_classification, error_slice_report
from app.analytics.ml_eval.columns import infer_prediction_columns
from app.analytics.ml_eval.probability import evaluate_prediction_scores
from app.analytics.ml_eval.regression import (
    evaluate_regression_or_forecast,
    summarise_existing_regression_metrics,
)


TaskHint = Literal["auto", "classification", "regression", "forecast", "scored_predictions"]


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() >= 0.90


def _infer_task_type(
        df: pd.DataFrame,
        actual_col: str | None,
        prediction_col: str | None,
        probability_col: str | None,
        task_hint: TaskHint,
) -> str:
    if task_hint != "auto":
        return task_hint

    if actual_col and prediction_col:
        actual_numeric = _is_numeric_series(df[actual_col])
        pred_numeric = _is_numeric_series(df[prediction_col])

        if actual_numeric and pred_numeric:
            unique_actual = pd.to_numeric(df[actual_col], errors="coerce").nunique(dropna=True)
            unique_pred = pd.to_numeric(df[prediction_col], errors="coerce").nunique(dropna=True)

            if unique_actual <= 20 and unique_pred <= 20:
                return "classification"

            return "regression"

        return "classification"

    if probability_col:
        return "scored_predictions"

    metric_like_cols = [
        c for c in df.columns
        if any(token in str(c).lower() for token in ["wmape", "wbias", "mae", "rmse", "mape"])
    ]

    if metric_like_cols:
        return "precomputed_regression_metrics"

    return "unknown"


def evaluate_ml_predictions(
        df: pd.DataFrame,
        actual_col: str | None = None,
        prediction_col: str | None = None,
        probability_col: str | None = None,
        id_col: str | None = None,
        task_hint: TaskHint = "auto",
        slice_cols: list[str] | None = None,
        top_n: int = 25,
) -> dict:
    """
    Evaluate common ML prediction datasets into engineering-grade metrics.

    Supports:
    - binary classification
    - multiclass classification
    - scored predictions without labels
    - regression / forecasting predictions
    - datasets that already contain metrics such as WMAPE/WBIAS
    """
    inferred = infer_prediction_columns(
        df=df,
        actual_col=actual_col,
        prediction_col=prediction_col,
        probability_col=probability_col,
        id_col=id_col,
    )

    task_type = _infer_task_type(
        df=df,
        actual_col=inferred.actual_col,
        prediction_col=inferred.prediction_col,
        probability_col=inferred.probability_col,
        task_hint=task_hint,
    )

    base = {
        "task_type": task_type,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": {
            "actual_col": inferred.actual_col,
            "prediction_col": inferred.prediction_col,
            "probability_col": inferred.probability_col,
            "id_col": inferred.id_col,
        },
    }

    if task_type == "classification":
        if not inferred.actual_col or not inferred.prediction_col:
            return {
                **base,
                "error": "Classification evaluation requires actual and prediction columns.",
            }

        evaluation = evaluate_classification(
            df=df,
            actual_col=inferred.actual_col,
            prediction_col=inferred.prediction_col,
            probability_col=inferred.probability_col,
        )

        slices = error_slice_report(
            df=df,
            actual_col=inferred.actual_col,
            prediction_col=inferred.prediction_col,
            slice_cols=slice_cols,
            top_n=top_n,
        )

        return {
            **base,
            "evaluation": evaluation,
            "worst_error_slices": slices,
            "engineering_readout": _classification_readout(evaluation),
        }

    if task_type in {"regression", "forecast"}:
        if not inferred.actual_col or not inferred.prediction_col:
            return {
                **base,
                "error": "Regression/forecast evaluation requires actual and prediction columns.",
            }

        evaluation = evaluate_regression_or_forecast(
            df=df,
            actual_col=inferred.actual_col,
            prediction_col=inferred.prediction_col,
        )

        return {
            **base,
            "evaluation": evaluation,
            "engineering_readout": _regression_readout(evaluation),
        }

    if task_type == "scored_predictions":
        if not inferred.probability_col:
            return {
                **base,
                "error": "Scored prediction evaluation requires a probability/score column.",
            }

        evaluation = evaluate_prediction_scores(
            df=df,
            probability_col=inferred.probability_col,
            prediction_col=inferred.prediction_col,
            id_col=inferred.id_col,
            top_n=top_n,
        )

        return {
            **base,
            "evaluation": evaluation,
            "engineering_readout": _scored_prediction_readout(evaluation),
        }

    if task_type == "precomputed_regression_metrics":
        evaluation = summarise_existing_regression_metrics(df)
        return {
            **base,
            "evaluation": evaluation,
            "engineering_readout": "Dataset appears to contain precomputed model quality metrics. Summarised metric distributions for monitoring and model comparison.",
        }

    return {
        **base,
        "error": (
            "Could not infer ML prediction structure. Provide actual_col, prediction_col, "
            "probability_col, or task_hint explicitly."
        ),
    }


def _classification_readout(evaluation: dict) -> str:
    accuracy = evaluation.get("accuracy")
    f1 = evaluation.get("f1_weighted")
    roc_auc = evaluation.get("roc_auc")

    parts = []

    if accuracy is not None:
        parts.append(f"Accuracy={accuracy:.4f}")

    if f1 is not None:
        parts.append(f"Weighted F1={f1:.4f}")

    if roc_auc is not None:
        parts.append(f"ROC AUC={roc_auc:.4f}")

    if not parts:
        return "Classification metrics computed. Inspect per-class precision, recall, F1, and confusion matrix."

    return "Classification evaluation complete: " + ", ".join(parts) + "."


def _regression_readout(evaluation: dict) -> str:
    wmape = evaluation.get("wmape")
    wbias = evaluation.get("wbias")
    rmse = evaluation.get("rmse")

    parts = []

    if wmape is not None:
        parts.append(f"WMAPE={wmape:.4f}")

    if wbias is not None:
        parts.append(f"WBIAS={wbias:.4f}")

    if rmse is not None:
        parts.append(f"RMSE={rmse:.4f}")

    if not parts:
        return "Regression/forecast metrics computed. Inspect absolute error, bias, and residual spread."

    return "Regression/forecast evaluation complete: " + ", ".join(parts) + "."


def _scored_prediction_readout(evaluation: dict) -> str:
    bands = evaluation.get("confidence_bands", {})
    high = bands.get("high_confidence_0_80_plus")

    if high is None:
        return "Prediction score distribution computed."

    return f"Scored prediction evaluation complete: {high} rows are high-confidence predictions at score >= 0.80."