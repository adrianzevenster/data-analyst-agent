"""Feature distribution drift detection between training and scoring data."""
from __future__ import annotations

import hashlib
import math

import pandas as pd


_MAX_COLS = 100
_MAX_CATEGORIES = 20
_PSI_BINS = 10
_PSI_MEDIUM = 0.1
_PSI_HIGH = 0.25


def _psi_numeric(bins: list[float], scoring: pd.Series) -> float:
    """Population Stability Index against training decile bins.

    Thresholds: PSI < 0.1 stable, 0.1–0.25 monitor, > 0.25 retrain.
    """
    s = scoring.dropna()
    if len(bins) < 2 or s.empty:
        return 0.0
    total = len(s)
    expected_pct = 1.0 / _PSI_BINS
    edges = list(bins)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    psi = 0.0
    for i in range(len(edges) - 1):
        actual_count = int(((s >= edges[i]) & (s < edges[i + 1])).sum())
        actual_pct = max(actual_count / total, 1e-6)
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
    return round(psi, 5)


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
            quantile_bins = [float(s.quantile(i / _PSI_BINS)) for i in range(_PSI_BINS + 1)]
            stats[col] = {
                "type": "numeric",
                "mean": float(s.mean()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "p5": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "missing_rate": missing_rate,
                "quantile_bins": quantile_bins,
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


def compute_fingerprint(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Lightweight data fingerprint: column-set hash + per-numeric distribution summary.

    Used to detect when scoring data has different columns or distributions from training.
    """
    col_hash = hashlib.md5("|".join(sorted(feature_cols)).encode()).hexdigest()[:12]
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if not s.empty:
                numeric_means[col] = round(float(s.mean()), 6)
                numeric_stds[col] = round(float(s.std()), 6) if len(s) > 1 else 0.0
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(feature_cols)),
        "column_hash": col_hash,
        "columns": sorted(feature_cols),
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
    }


def compare_fingerprints(scoring_df: pd.DataFrame, feature_cols: list[str], stored: dict) -> dict:
    """Compare current scoring data against the stored training fingerprint.

    Returns a lineage report: whether columns changed and whether numeric
    feature distributions have shifted significantly (mean shift > 2 train-σ).
    """
    if not stored:
        return {"lineage_ok": True, "message": "No fingerprint stored for this model."}

    stored_cols = set(stored.get("columns", []))
    current_cols = set(feature_cols)
    added = sorted(current_cols - stored_cols)
    removed = sorted(stored_cols - current_cols)
    col_hash_match = (
        stored.get("column_hash")
        == hashlib.md5("|".join(sorted(feature_cols)).encode()).hexdigest()[:12]
    )

    stored_means = stored.get("numeric_means", {})
    stored_stds = stored.get("numeric_stds", {})
    shifted: list[str] = []
    for col in feature_cols:
        if col not in scoring_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(scoring_df[col]):
            continue
        s_mean = stored_means.get(col)
        s_std = stored_stds.get(col)
        if s_mean is None or not s_std:
            continue
        s = scoring_df[col].dropna()
        if s.empty:
            continue
        shift = abs(float(s.mean()) - s_mean) / s_std
        if shift > 2.0:
            shifted.append(col)

    lineage_ok = col_hash_match and not shifted
    return {
        "lineage_ok": lineage_ok,
        "col_hash_match": col_hash_match,
        "columns_added": added,
        "columns_removed": removed,
        "distribution_shifted": shifted,
        "training_n_rows": stored.get("n_rows"),
    }


def detect_drift_tool(
    df: pd.DataFrame,
    model_id: str,
    model_manager=None,
) -> dict:
    """Standalone drift and schema check for the agent tool registry.

    Loads the model's stored training stats and compares them against *df*.
    Returns a drift report, schema diff (missing/extra features), and a lineage
    summary without running any scoring.
    """
    from app.analytics.ml_train.model_store import ModelManager

    manager = model_manager or ModelManager()
    try:
        _, meta = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    expected = meta.feature_cols
    actual_set = set(df.columns)
    missing_features = [c for c in expected if c not in actual_set]
    extra_features = sorted(actual_set - set(expected) - {meta.target_col})
    avail = [c for c in expected if c in actual_set]

    training_stats: dict = getattr(meta, "training_stats", None) or {}
    drift_report: dict = {
        "drifted_features": [],
        "n_drifted": 0,
        "n_features_checked": 0,
        "drift_rate": 0.0,
        "overall_severity": "none",
    }
    if training_stats and avail:
        try:
            drift_report = check_drift(df[avail], training_stats)
        except Exception:
            pass

    lineage: dict | None = None
    stored_fp = getattr(meta, "data_fingerprint", None)
    if stored_fp and avail:
        try:
            lineage = compare_fingerprints(df[avail], avail, stored_fp)
        except Exception:
            pass

    severity = drift_report.get("overall_severity", "none")
    n_drifted = drift_report.get("n_drifted", 0)
    n_checked = drift_report.get("n_features_checked", 0)

    charts: list[dict] = []
    drifted_features = drift_report.get("drifted_features", [])
    if drifted_features:
        chart_rows = [
            {
                "feature": f["feature"][:24],
                "psi": round(f.get("psi", 0.0), 4) if f.get("psi") is not None
                       else round(min(f.get("mean_shift_std", 0.0) / 5.0, 1.0), 4),
            }
            for f in drifted_features[:12]
        ]
        charts.append({
            "type": "bar",
            "title": f"Drift severity by feature (model '{model_id[:8]}…')",
            "x": "feature",
            "y": "psi",
            "data": chart_rows,
        })

    return {
        "model_id": model_id,
        "target_col": meta.target_col,
        "task_type": meta.task_type,
        "n_rows": int(len(df)),
        "n_features_checked": n_checked,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "drift": drift_report,
        "lineage": lineage,
        "charts": charts,
        "engineering_readout": (
            f"Drift check for model '{model_id[:8]}' on {len(df)} rows: "
            f"{n_drifted}/{n_checked} features drifted (severity: {severity})."
            + (f" Missing features: {missing_features}." if missing_features else "")
            + (f" Extra columns ignored: {extra_features[:5]}." if extra_features else "")
        ),
    }


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

            # PSI is the primary severity signal when training quantile bins are available.
            # Falls back to 3-sigma heuristic for models trained before quantile_bins were stored.
            psi: float | None = None
            bins = stat.get("quantile_bins")
            if bins and len(bins) == _PSI_BINS + 1:
                psi = _psi_numeric(bins, s)
                triggered = psi >= _PSI_MEDIUM or missing_delta > 0.2
                severity = "high" if (psi >= _PSI_HIGH or missing_delta > 0.4) else "medium"
            else:
                triggered = mean_shift > 3 or std_ratio > 3 or missing_delta > 0.2
                severity = "high" if (mean_shift > 5 or std_ratio > 5 or missing_delta > 0.4) else "medium"

            if triggered:
                entry: dict = {
                    "feature": col,
                    "type": "numeric",
                    "mean_shift_std": round(mean_shift, 2),
                    "std_ratio": round(std_ratio, 2),
                    "missing_rate_delta": round(missing_delta, 3),
                    "severity": severity,
                }
                if psi is not None:
                    entry["psi"] = psi
                drifted.append(entry)

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
