from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.drift import check_drift
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import engineer_lag_features


def score_with_model(
    df: pd.DataFrame,
    model_id: str,
    top_n: int = 500,
    model_manager: ModelManager | None = None,
) -> dict:
    manager = model_manager or ModelManager()
    pipeline, meta = manager.load_model(model_id)

    # Re-apply lag/rolling features when the model was trained with them.
    lag_config = getattr(meta, "lag_config", None)
    if lag_config:
        df, _ = engineer_lag_features(
            df,
            lag_config["sort_col"],
            lag_config["lag_cols"],
            lags=lag_config.get("lags", [1, 7]),
            windows=lag_config.get("windows", [7]),
        )

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns required by model {model_id}: {missing}")

    X = df[meta.feature_cols]
    predictions = pipeline.predict(X)

    if getattr(meta, "log_transform_target", False):
        predictions = np.expm1(predictions)

    out = df.copy()
    out["prediction"] = predictions

    halfwidth = getattr(meta, "conformal_halfwidth", None)
    if halfwidth is not None and meta.task_type == "regression":
        lower = predictions - halfwidth
        if getattr(meta, "log_transform_target", False):
            lower = lower.clip(min=0)
        out["prediction_lower_90"] = lower
        out["prediction_upper_90"] = predictions + halfwidth

    if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
        # CalibratedClassifierCV exposes .classes_ directly; unwrap plain Pipeline
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(classes) == 2:
            probs = pipeline.predict_proba(X)[:, -1]
            out["prediction_probability"] = probs
            threshold = getattr(meta, "optimal_threshold", None) or 0.5
            if threshold != 0.5:
                out["prediction"] = np.where(probs >= threshold, classes[-1], classes[0])

    n_rows = len(out)
    scored_rows = out.head(top_n).reset_index(drop=True).to_dict(orient="records")

    drift_report: dict | None = None
    training_stats = getattr(meta, "training_stats", None)
    if training_stats:
        try:
            drift_report = check_drift(X, training_stats)
        except Exception:
            pass

    pi_note = (
        f"90% prediction intervals included (±{halfwidth:.4f})."
        if halfwidth is not None and meta.task_type == "regression"
        else None
    )

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "target_col": meta.target_col,
        "n_rows_scored": n_rows,
        "scored_rows": scored_rows,
        "drift": drift_report,
        "conformal_halfwidth": halfwidth,
        "engineering_readout": (
            f"Scored {n_rows} rows with model {model_id} ({meta.model_type}, {meta.task_type})."
            + (f" {pi_note}" if pi_note else "")
        ),
    }
