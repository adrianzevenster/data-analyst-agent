from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.model_store import ModelManager


def compute_pdp(
    df: pd.DataFrame,
    model_id: str,
    feature_cols: list[str] | None = None,
    n_top_features: int = 5,
    grid_resolution: int = 20,
    model_manager: ModelManager | None = None,
) -> dict:
    """Compute partial dependence plots for a stored model.

    Varies each feature from its 5th to 95th percentile while holding all
    others at observed values, averaged over a subsample of rows.  Returns
    a ``charts`` list so the executor renders them in the Latest-query section.
    """
    manager = model_manager or ModelManager()
    try:
        meta = manager.get_meta(model_id)
        pipeline, _ = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Dataset is missing required model feature(s): {', '.join(missing[:5])}"}

    # Determine which features to plot
    if feature_cols:
        plot_features = [f for f in feature_cols if f in meta.feature_cols and f in df.columns]
    else:
        feat_imp = (meta.evaluation or {}).get("feature_importance", [])
        if feat_imp:
            ranked = sorted(feat_imp, key=lambda x: x.get("importance", 0), reverse=True)
            candidates = [f["feature"] for f in ranked]
        else:
            candidates = meta.feature_cols
        plot_features = [f for f in candidates if f in df.columns][:n_top_features]

    if not plot_features:
        return {"error": "No matching features found to plot."}

    X = df[meta.feature_cols].copy()
    n_sample = min(200, len(X))
    X_sample = X.sample(n=n_sample, random_state=42) if len(X) > n_sample else X.copy()

    charts: list[dict] = []
    skipped: list[str] = []

    for feat in plot_features[:6]:
        col = X[feat]
        is_numeric = pd.api.types.is_numeric_dtype(col)

        if is_numeric:
            lo, hi = float(col.quantile(0.05)), float(col.quantile(0.95))
            if abs(hi - lo) < 1e-10:
                skipped.append(feat)
                continue
            grid: list = np.linspace(lo, hi, grid_resolution).tolist()
        else:
            grid = col.value_counts().head(grid_resolution).index.tolist()
            if not grid:
                skipped.append(feat)
                continue

        effects: list[float] = []
        for val in grid:
            X_pert = X_sample.copy()
            X_pert[feat] = val
            try:
                if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_pert)
                    eff = float(np.mean(probs[:, 1])) if probs.shape[1] == 2 else float(np.mean(np.max(probs, axis=1)))
                else:
                    eff = float(np.mean(pipeline.predict(X_pert)))
                effects.append(eff)
            except Exception:
                continue

        if len(effects) < 2:
            skipped.append(feat)
            continue

        grid = grid[:len(effects)]
        charts.append({
            "type": "line",
            "title": f"PDP: {feat}",
            "x": "value",
            "y": "effect",
            "data": [
                {
                    "value": round(float(v), 4) if is_numeric else str(v),
                    "effect": round(e, 6),
                }
                for v, e in zip(grid, effects)
            ],
        })

    task_label = (
        "predicted probability (positive class)"
        if meta.task_type == "classification"
        else "predicted value"
    )
    readout = (
        f"Partial dependence plots for {len(charts)} feature(s) of model {model_id[:8]} "
        f"(target: '{meta.target_col}'). Each curve shows {task_label} as the feature "
        f"varies from P5 to P95, averaged over {n_sample} samples."
    )
    if skipped:
        readout += f" Skipped {len(skipped)} feature(s) with insufficient range: {', '.join(skipped)}."

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "target_col": meta.target_col,
        "feature_cols": plot_features,
        "n_features_plotted": len(charts),
        "sample_size": n_sample,
        "charts": charts,
        "engineering_readout": readout,
    }


def compute_ice(
    df: pd.DataFrame,
    model_id: str,
    feature_col: str | None = None,
    n_rows: int = 20,
    grid_resolution: int = 20,
    model_manager: ModelManager | None = None,
) -> dict:
    """Compute Individual Conditional Expectation (ICE) curves for a single feature.

    Unlike PDP (which averages), ICE shows one curve per row, revealing heterogeneous
    feature effects — e.g. income has a positive effect on churn for young customers
    but negative for older ones.
    """
    manager = model_manager or ModelManager()
    try:
        meta = manager.get_meta(model_id)
        pipeline, _ = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Dataset missing required feature(s): {', '.join(missing[:5])}"}

    # Pick feature: explicit > top by importance > first numeric
    feat: str | None = None
    if feature_col:
        if feature_col not in meta.feature_cols:
            return {"error": f"Feature '{feature_col}' not in model features."}
        if feature_col not in df.columns:
            return {"error": f"Feature '{feature_col}' not in dataset."}
        feat = feature_col
    else:
        feat_imp = (meta.evaluation or {}).get("feature_importance", [])
        numeric_feats = [
            c for c in meta.feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if feat_imp:
            ranked = [f["feature"] for f in sorted(feat_imp, key=lambda x: x.get("importance", 0), reverse=True)]
            feat = next((f for f in ranked if f in numeric_feats), numeric_feats[0] if numeric_feats else None)
        else:
            feat = numeric_feats[0] if numeric_feats else None

    if feat is None:
        return {"error": "No numeric feature available for ICE plot."}

    col = df[feat]
    if not pd.api.types.is_numeric_dtype(col):
        return {"error": f"Feature '{feat}' is not numeric. ICE plots require a numeric feature."}

    lo, hi = float(col.quantile(0.05)), float(col.quantile(0.95))
    if abs(hi - lo) < 1e-10:
        return {"error": f"Feature '{feat}' has no range (P5==P95)."}

    grid = np.linspace(lo, hi, grid_resolution).tolist()

    n_sample = min(n_rows, len(df))
    X_sample = df[meta.feature_cols].sample(n=n_sample, random_state=42) if len(df) > n_sample else df[meta.feature_cols].copy()
    row_keys = [f"r{i}" for i in range(n_sample)]

    # Build data points: one dict per grid value with a key per row
    data: list[dict] = []
    for val in grid:
        point: dict = {"value": round(float(val), 4)}
        X_pert = X_sample.copy()
        X_pert[feat] = val
        try:
            if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_pert)
                preds = probs[:, 1].tolist() if probs.shape[1] == 2 else probs.max(axis=1).tolist()
            else:
                preds = pipeline.predict(X_pert).tolist()
            for key, pred in zip(row_keys, preds):
                point[key] = round(float(pred), 6)
        except Exception:
            continue
        data.append(point)

    if not data:
        return {"error": f"ICE computation produced no data for feature '{feat}'."}

    task_label = "probability" if meta.task_type == "classification" else "predicted value"
    chart = {
        "type": "line",
        "title": f"ICE: {feat} — {n_sample} individual curves",
        "x": "value",
        "y_series": row_keys,
        "data": data,
    }

    readout = (
        f"ICE plot for feature '{feat}' of model {model_id[:8]} "
        f"(target: '{meta.target_col}'). Shows {task_label} for {n_sample} individual rows "
        f"as '{feat}' varies from {lo:.3g} to {hi:.3g}. "
        "Spread of curves indicates heterogeneous feature effects across rows."
    )

    return {
        "model_id": model_id,
        "target_col": meta.target_col,
        "task_type": meta.task_type,
        "feature_col": feat,
        "n_rows_plotted": n_sample,
        "charts": [chart],
        "engineering_readout": readout,
    }
