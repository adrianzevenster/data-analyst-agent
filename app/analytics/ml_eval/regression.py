from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_or_forecast(
        df: pd.DataFrame,
        actual_col: str,
        prediction_col: str,
) -> dict:
    d = df[[actual_col, prediction_col]].copy()
    d[actual_col] = pd.to_numeric(d[actual_col], errors="coerce")
    d[prediction_col] = pd.to_numeric(d[prediction_col], errors="coerce")
    d = d.dropna()

    if d.empty:
        return {
            "task_type": "regression",
            "error": "No numeric non-null rows available for regression evaluation.",
        }

    y_true = d[actual_col]
    y_pred = d[prediction_col]
    error = y_pred - y_true
    abs_error = error.abs()

    denominator = float(y_true.abs().sum())
    wmape = float(abs_error.sum() / denominator) if denominator else None
    wbias = float(error.sum() / denominator) if denominator else None

    result: dict = {
        "task_type": "regression_or_forecast",
        "n_rows_evaluated": int(len(d)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float((abs_error / y_true.replace(0, np.nan).abs()).mean()),
        "wmape": wmape,
        "wbias": wbias,
        "r2": float(r2_score(y_true, y_pred)) if len(d) > 1 else None,
        "actual_sum": float(y_true.sum()),
        "prediction_sum": float(y_pred.sum()),
        "mean_error": float(error.mean()),
        "median_absolute_error": float(abs_error.median()),
        "p90_absolute_error": float(abs_error.quantile(0.90)),
        "p95_absolute_error": float(abs_error.quantile(0.95)),
    }

    # Actual vs predicted scatter (max 200 points)
    try:
        _n = min(len(y_true), 200)
        _idx = np.linspace(0, len(y_true) - 1, _n, dtype=int)
        _corr = float(np.corrcoef(y_true.values, y_pred.values)[0, 1]) if len(y_true) > 1 else None
        result["actual_vs_predicted"] = {
            "type": "scatter",
            "title": f"Actual vs Predicted{f'  (r={_corr:.3f})' if _corr is not None else ''}",
            "x": "actual",
            "y": "prediction",
            "data": [
                {"actual": round(float(y_true.iloc[i]), 6), "prediction": round(float(y_pred.iloc[i]), 6)}
                for i in _idx
            ],
        }
    except Exception:
        pass

    # Residuals histogram
    try:
        residuals = error.values
        _bins = min(25, max(10, int(np.sqrt(len(residuals)))))
        counts, edges = np.histogram(residuals, bins=_bins)
        result["residuals_hist"] = {
            "type": "histogram",
            "title": "Residual Distribution  (prediction − actual)",
            "column": "residual",
            "data": [
                {"bin_label": f"{edges[i]:.3g} – {edges[i + 1]:.3g}", "count": int(counts[i])}
                for i in range(len(counts))
            ],
        }
    except Exception:
        pass

    return result


def summarise_existing_regression_metrics(df: pd.DataFrame) -> dict:
    metric_cols = [
        c for c in df.columns
        if any(token in str(c).lower() for token in ["wmape", "wbias", "mae", "rmse", "mape"])
    ]

    if not metric_cols:
        return {
            "task_type": "precomputed_regression_metrics",
            "error": "No common regression metric columns found.",
        }

    summary: dict[str, object] = {
        "task_type": "precomputed_regression_metrics",
        "n_rows": int(len(df)),
        "metric_columns": metric_cols,
        "metrics": {},
    }
    metrics_dict: dict[str, object] = {}

    for col in metric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        metrics_dict[col] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
        }

    summary["metrics"] = metrics_dict
    return summary