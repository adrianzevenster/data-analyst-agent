from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.drift import check_drift, compare_fingerprints
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

    # Regression: conformal prediction intervals
    halfwidth = getattr(meta, "conformal_halfwidth", None)
    if halfwidth is not None and meta.task_type == "regression":
        lower = predictions - halfwidth
        if getattr(meta, "log_transform_target", False):
            lower = lower.clip(min=0)
        out["prediction_lower_90"] = lower
        out["prediction_upper_90"] = predictions + halfwidth

    # Classification: probabilities + conformal prediction sets
    conf_clf_threshold = getattr(meta, "conformal_classification_threshold", None)
    pred_set_info: dict | None = None
    if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))

        if len(classes) >= 2:
            probs = pipeline.predict_proba(X)

            # Binary probability column
            if len(classes) == 2:
                prob_col = probs[:, -1]
                out["prediction_probability"] = prob_col
                threshold = getattr(meta, "optimal_threshold", None) or 0.5
                if threshold != 0.5:
                    out["prediction"] = np.where(prob_col >= threshold, classes[-1], classes[0])

            # Conformal prediction sets
            if conf_clf_threshold is not None:
                pred_sets = []
                for row_probs in probs:
                    ps = [str(classes[i]) for i, p in enumerate(row_probs) if p >= 1.0 - conf_clf_threshold]
                    pred_sets.append(
                        "|".join(sorted(ps)) if ps else str(classes[int(np.argmax(row_probs))])
                    )
                out["prediction_set"] = pred_sets
                avg_set_size = float(np.mean([len(s.split("|")) for s in pred_sets]))
                pred_set_info = {
                    "coverage_target": 0.90,
                    "threshold": round(conf_clf_threshold, 4),
                    "avg_set_size": round(avg_set_size, 2),
                    "n_singleton": int(sum(1 for s in pred_sets if "|" not in s)),
                }

    n_rows = len(out)
    scored_rows = out.head(top_n).reset_index(drop=True).to_dict(orient="records")

    drift_report: dict | None = None
    training_stats = getattr(meta, "training_stats", None)
    if training_stats:
        try:
            drift_report = check_drift(X, training_stats)
        except Exception:
            pass

    # Data lineage: compare scoring column set and distributions to training fingerprint
    lineage_report: dict | None = None
    stored_fp = getattr(meta, "data_fingerprint", None)
    if stored_fp:
        try:
            lineage_report = compare_fingerprints(X, meta.feature_cols, stored_fp)
        except Exception:
            pass

    pi_note = (
        f"90% prediction intervals included (±{halfwidth:.4f})."
        if halfwidth is not None and meta.task_type == "regression"
        else None
    )
    ps_note = (
        f"Conformal prediction sets (90% coverage) added as prediction_set column — avg size {pred_set_info['avg_set_size']:.1f}."
        if pred_set_info
        else None
    )

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "target_col": meta.target_col,
        "n_rows_scored": n_rows,
        "scored_rows": scored_rows,
        "drift": drift_report,
        "lineage": lineage_report,
        "prediction_set_info": pred_set_info,
        "conformal_halfwidth": halfwidth,
        "engineering_readout": (
            f"Scored {n_rows} rows with model {model_id} ({meta.model_type}, {meta.task_type})."
            + (f" {pi_note}" if pi_note else "")
            + (f" {ps_note}" if ps_note else "")
        ),
    }
