from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.analytics.ml_train.model_store import ModelManager


def explain_model(
    df: pd.DataFrame,
    model_id: str,
    sample: int = 500,
    n_repeats: int = 10,
    model_manager: ModelManager | None = None,
) -> dict:
    """Permutation feature importance for a stored model evaluated on the current dataset."""
    manager = model_manager or ModelManager()
    try:
        pipeline, meta = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Dataset missing model features: {', '.join(missing)}"}
    if meta.target_col not in df.columns:
        return {"error": f"Target column '{meta.target_col}' not found in dataset."}

    d = df[meta.feature_cols + [meta.target_col]].dropna(subset=[meta.target_col])
    if d.empty:
        return {"error": "No rows remain after dropping nulls on target column."}
    if len(d) > sample:
        d = d.sample(n=sample, random_state=42)

    X = d[meta.feature_cols]
    y = d[meta.target_col]

    if meta.log_transform_target and meta.task_type == "regression":
        y = np.log1p(pd.to_numeric(y, errors="coerce").fillna(0).astype(float))

    scoring = "f1_weighted" if meta.task_type == "classification" else "r2"

    try:
        perm = permutation_importance(
            pipeline, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )
    except Exception as exc:
        return {"error": f"Permutation importance computation failed: {exc}"}

    importances = sorted(
        [
            {
                "feature": meta.feature_cols[i],
                "importance_mean": round(float(perm.importances_mean[i]), 6),
                "importance_std": round(float(perm.importances_std[i]), 6),
            }
            for i in range(len(meta.feature_cols))
        ],
        key=lambda x: -cast(float, x["importance_mean"]),
    )

    top = importances[:15]
    negative_count = sum(1 for f in importances if cast(float, f["importance_mean"]) < 0)
    top_name = top[0]["feature"] if top else "n/a"
    top_delta = top[0]["importance_mean"] if top else 0.0

    noise_note = f" {negative_count} feature(s) had negative importance (noise)." if negative_count else ""

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "n_samples": int(len(d)),
        "scoring_metric": scoring,
        "feature_importances": top,
        "negative_importance_count": negative_count,
        "engineering_readout": (
            f"Permutation importance ({scoring}) for {meta.model_type} predicting "
            f"'{meta.target_col}' on {len(d)} samples. "
            f"Top feature: '{top_name}' (mean Δ={top_delta:+.4f}).{noise_note}"
        ),
    }
