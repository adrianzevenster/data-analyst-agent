"""Naive baseline models for sanity-checking whether a trained model adds value."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score


def _wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def compute_baselines(
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    log_transform_target: bool = False,
) -> dict:
    """
    Fit naive baselines on y_train, evaluate on y_test.

    Returns a dict with:
      - baselines: per-strategy metrics
      - best_baseline_metric: best baseline value on the primary metric
      - model_primary_metric_name: which metric to compare against
      - beats_baseline: None (filled by caller after the main model trains)
      - delta: None (filled by caller)
    """
    if task_type == "classification":
        results: dict[str, dict] = {}
        for strategy in ("most_frequent", "stratified"):
            try:
                dummy = DummyClassifier(strategy=strategy, random_state=42)
                dummy.fit(np.zeros((len(y_train), 1)), y_train)
                y_pred = dummy.predict(np.zeros((len(y_test), 1)))
                results[strategy] = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1_weighted": round(
                        float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
                    ),
                }
            except Exception:
                pass

        best_baseline = max(
            (v.get("accuracy", 0.0) for v in results.values()), default=0.0
        )
        return {
            "baselines": results,
            "primary_metric": "accuracy",
            "best_baseline_metric": round(best_baseline, 4),
            "beats_baseline": None,
            "delta": None,
        }

    else:  # regression
        results_reg: dict[str, dict] = {}
        y_tr = y_train.to_numpy().astype(float)
        y_te = y_test.to_numpy().astype(float)

        for strategy in ("mean", "median"):
            try:
                dummy = DummyRegressor(strategy=strategy)
                dummy.fit(np.zeros((len(y_tr), 1)), y_tr)
                y_pred = dummy.predict(np.zeros((len(y_te), 1)))
                if log_transform_target:
                    y_pred_eval = np.expm1(y_pred)
                    y_te_eval = np.expm1(y_te)
                else:
                    y_pred_eval, y_te_eval = y_pred, y_te
                results_reg[strategy] = {
                    "wmape": round(_wmape(y_te_eval, y_pred_eval), 4),
                    "r2": round(
                        float(1 - np.sum((y_te_eval - y_pred_eval) ** 2) / max(np.sum((y_te_eval - np.mean(y_te_eval)) ** 2), 1e-9)),
                        4,
                    ),
                }
            except Exception:
                pass

        # Lower WMAPE = better baseline (harder to beat)
        best_baseline = min(
            (v.get("wmape", float("inf")) for v in results_reg.values()),
            default=float("inf"),
        )
        return {
            "baselines": results_reg,
            "primary_metric": "wmape",
            "best_baseline_metric": round(best_baseline, 4) if best_baseline != float("inf") else None,
            "beats_baseline": None,
            "delta": None,
        }


def finalise_baseline_comparison(
    baseline: dict,
    model_metric_value: float | None,
) -> dict:
    """Attach the model's metric and compute beats_baseline + delta."""
    b = dict(baseline)
    b["model_metric"] = model_metric_value
    if model_metric_value is None or b.get("best_baseline_metric") is None:
        return b

    if b["primary_metric"] == "accuracy":
        # Higher is better
        delta = round(float(model_metric_value) - float(b["best_baseline_metric"]), 4)
        beats = model_metric_value > b["best_baseline_metric"]
    else:
        # WMAPE: lower is better; positive delta means model is better
        delta = round(float(b["best_baseline_metric"]) - float(model_metric_value), 4)
        beats = model_metric_value < b["best_baseline_metric"]

    b["delta"] = delta
    b["beats_baseline"] = beats
    return b
