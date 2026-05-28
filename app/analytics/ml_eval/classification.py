from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def _to_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_classification(
        df: pd.DataFrame,
        actual_col: str,
        prediction_col: str,
        probability_col: str | None = None,
) -> dict:
    d = df[[actual_col, prediction_col] + ([probability_col] if probability_col else [])].dropna()

    if d.empty:
        return {
            "task_type": "classification",
            "error": "No non-null rows available for classification evaluation.",
        }

    y_true = d[actual_col]
    y_pred = d[prediction_col]
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)).tolist())

    metrics = {
        "task_type": "classification",
        "n_rows_evaluated": int(len(d)),
        "n_classes": int(len(labels)),
        "labels": [_to_jsonable(v) for v in labels],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": {
            "labels": [_to_jsonable(v) for v in labels],
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        },
    }

    if probability_col:
        y_score = pd.to_numeric(d[probability_col], errors="coerce")

        if y_score.notna().any():
            metrics["probability"] = {
                "column": probability_col,
                "mean": float(y_score.mean()),
                "p50": float(y_score.quantile(0.50)),
                "p90": float(y_score.quantile(0.90)),
                "p95": float(y_score.quantile(0.95)),
                "p99": float(y_score.quantile(0.99)),
            }

            unique_true = pd.Series(y_true).nunique(dropna=True)

            if unique_true == 2:
                y_true_numeric = pd.to_numeric(y_true, errors="coerce")
                if y_true_numeric.notna().all():
                    metrics["roc_auc"] = float(roc_auc_score(y_true_numeric, y_score))
                    metrics["average_precision"] = float(average_precision_score(y_true_numeric, y_score))

    return metrics


def error_slice_report(
        df: pd.DataFrame,
        actual_col: str,
        prediction_col: str,
        slice_cols: list[str] | None = None,
        top_n: int = 20,
) -> list[dict]:
    if not slice_cols:
        slice_cols = [
                         c for c in df.columns
                         if c not in {actual_col, prediction_col}
                            and not pd.api.types.is_numeric_dtype(df[c])
                     ][:3]

    rows: list[dict] = []

    d = df.copy()
    d["_is_correct"] = d[actual_col] == d[prediction_col]

    for col in slice_cols:
        if col not in d.columns:
            continue

        grouped = (
            d.groupby(col, dropna=False)
            .agg(
                n=(actual_col, "size"),
                accuracy=("_is_correct", "mean"),
            )
            .reset_index()
        )

        grouped["error_rate"] = 1.0 - grouped["accuracy"]
        grouped = grouped.sort_values(["error_rate", "n"], ascending=[False, False]).head(top_n)

        for _, r in grouped.iterrows():
            rows.append(
                {
                    "slice_column": col,
                    "slice_value": _to_jsonable(r[col]),
                    "n": int(r["n"]),
                    "accuracy": float(r["accuracy"]),
                    "error_rate": float(r["error_rate"]),
                }
            )

    return rows