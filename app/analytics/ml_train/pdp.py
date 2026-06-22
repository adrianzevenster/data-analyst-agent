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
