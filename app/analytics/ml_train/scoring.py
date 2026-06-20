from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.model_store import ModelManager


def score_with_model(
    df: pd.DataFrame,
    model_id: str,
    top_n: int = 500,
    model_manager: ModelManager | None = None,
) -> dict:
    manager = model_manager or ModelManager()
    pipeline, meta = manager.load_model(model_id)

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns required by model {model_id}: {missing}")

    X = df[meta.feature_cols]
    predictions = pipeline.predict(X)

    if getattr(meta, "log_transform_target", False):
        predictions = np.expm1(predictions)

    out = df.copy()
    out["prediction"] = predictions

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

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "target_col": meta.target_col,
        "n_rows_scored": n_rows,
        "scored_rows": scored_rows,
        "engineering_readout": (
            f"Scored {n_rows} rows with model {model_id} ({meta.model_type}, {meta.task_type})."
        ),
    }
