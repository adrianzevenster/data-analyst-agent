"""Feature distribution drift detection between training and scoring data."""
from __future__ import annotations

import pandas as pd


_MAX_COLS = 100
_MAX_CATEGORIES = 20


def compute_training_stats(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Capture per-feature summary statistics from the training split.
    Stored in ModelMeta.training_stats and used later to detect drift at scoring time.
    """
    stats: dict[str, dict] = {}
    all_cols = (numeric_cols + categorical_cols)[:_MAX_COLS]

    for col in all_cols:
        if col not in X_train.columns:
            continue
        missing_rate = float(X_train[col].isna().mean())

        if col in numeric_cols and pd.api.types.is_numeric_dtype(X_train[col]):
            s = X_train[col].dropna()
            if s.empty:
                continue
            stats[col] = {
                "type": "numeric",
                "mean": float(s.mean()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "p5": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "missing_rate": missing_rate,
            }
        else:
            vc = X_train[col].astype(str).value_counts(normalize=True)
            stats[col] = {
                "type": "categorical",
                "top_categories": {str(k): round(float(v), 4) for k, v in vc.head(_MAX_CATEGORIES).items()},
                "n_unique": int(X_train[col].nunique(dropna=True)),
                "missing_rate": missing_rate,
            }

    return stats


def check_drift(X: pd.DataFrame, training_stats: dict) -> dict:
    """
    Compare X against training-time statistics and return a drift report.

    Returns:
      drifted_features: list of per-feature drift dicts (sorted severity desc)
      n_drifted: int
      n_features_checked: int
      drift_rate: fraction of checked features that drifted
      overall_severity: "none" | "medium" | "high"
    """
    drifted: list[dict] = []

    for col, stat in training_stats.items():
        if col not in X.columns:
            continue

        if stat["type"] == "numeric":
            s = X[col].dropna()
            if s.empty:
                continue
            curr_mean = float(s.mean())
            curr_std = float(s.std()) if len(s) > 1 else 0.0
            curr_missing = float(X[col].isna().mean())

            train_std = stat["std"]
            mean_shift = abs(curr_mean - stat["mean"]) / max(train_std, 1e-9)
            std_ratio = max(curr_std, train_std) / max(min(curr_std, train_std), 1e-9) if min(curr_std, train_std) > 0 else 1.0
            missing_delta = abs(curr_missing - stat["missing_rate"])

            if mean_shift > 3 or std_ratio > 3 or missing_delta > 0.2:
                severity = "high" if (mean_shift > 5 or std_ratio > 5 or missing_delta > 0.4) else "medium"
                drifted.append({
                    "feature": col,
                    "type": "numeric",
                    "mean_shift_std": round(mean_shift, 2),
                    "std_ratio": round(std_ratio, 2),
                    "missing_rate_delta": round(missing_delta, 3),
                    "severity": severity,
                })

        elif stat["type"] == "categorical":
            known = set(stat["top_categories"].keys())
            curr_vals = X[col].dropna().astype(str)
            if curr_vals.empty:
                continue
            n_new = curr_vals.isin(known).__invert__().sum()
            new_rate = float(n_new / len(curr_vals))
            curr_missing = float(X[col].isna().mean())
            missing_delta = abs(curr_missing - stat["missing_rate"])

            if new_rate > 0.1 or missing_delta > 0.2:
                severity = "high" if (new_rate > 0.3 or missing_delta > 0.4) else "medium"
                drifted.append({
                    "feature": col,
                    "type": "categorical",
                    "new_category_rate": round(new_rate, 3),
                    "missing_rate_delta": round(missing_delta, 3),
                    "severity": severity,
                })

    drifted.sort(key=lambda f: 0 if f["severity"] == "high" else 1)
    n_checked = sum(1 for c in training_stats if c in X.columns)

    return {
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "n_features_checked": n_checked,
        "drift_rate": round(len(drifted) / max(n_checked, 1), 3),
        "overall_severity": (
            "high" if any(f["severity"] == "high" for f in drifted)
            else "medium" if drifted
            else "none"
        ),
    }
