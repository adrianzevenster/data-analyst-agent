from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.model_store import ModelManager


def what_if_predict(
    df: pd.DataFrame,
    model_id: str,
    row_idx: int = 0,
    overrides: dict | None = None,
    model_manager: ModelManager | None = None,
) -> dict:
    """Apply column-value overrides to a single row and return the model's new prediction.

    Shows how changing specific feature values shifts the predicted outcome, making
    it easy to answer 'what would the model predict if income were 80000?'
    """
    manager = model_manager or ModelManager()
    try:
        meta = manager.get_meta(model_id)
        pipeline, _ = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    if row_idx < 0 or row_idx >= len(df):
        return {"error": f"row_idx {row_idx} out of range (dataset has {len(df)} rows)."}

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Dataset missing required feature(s): {', '.join(missing[:5])}"}

    overrides = overrides or {}
    invalid = [k for k in overrides if k not in meta.feature_cols]
    if invalid:
        return {"error": f"Override key(s) not in model features: {', '.join(invalid[:5])}"}

    if not overrides:
        return {
            "error": (
                "No overrides provided. Supply a dict of {column: new_value} pairs, "
                "e.g. overrides={'income': 80000}."
            )
        }

    original_row = df.iloc[[row_idx]][meta.feature_cols].copy()
    modified_row = original_row.copy()
    for col, val in overrides.items():
        modified_row[col] = val

    def _predict_row(X: pd.DataFrame) -> dict:
        if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)
            cls_idx = int(np.argmax(probs[0]))
            pred_class = pipeline.classes_[cls_idx]
            prob = float(probs[0, 1]) if probs.shape[1] == 2 else float(probs[0].max())
            return {"class": str(pred_class), "probability": round(prob, 4)}
        pred = float(pipeline.predict(X)[0])
        result: dict = {"value": round(pred, 4)}
        if meta.conformal_halfwidth is not None:
            hw = meta.conformal_halfwidth
            result["lower_90"] = round(pred - hw, 4)
            result["upper_90"] = round(pred + hw, 4)
        return result

    try:
        original_pred = _predict_row(original_row)
        new_pred = _predict_row(modified_row)
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    if meta.task_type == "classification":
        orig_p = original_pred.get("probability", 0.0)
        new_p = new_pred.get("probability", 0.0)
        delta = round(new_p - orig_p, 4)
        if original_pred["class"] == new_pred["class"]:
            delta_str = f"class unchanged ({original_pred['class']}), probability {delta:+.4f}"
        else:
            delta_str = f"class flipped {original_pred['class']} → {new_pred['class']}, probability {delta:+.4f}"
    else:
        delta = round(new_pred["value"] - original_pred["value"], 4)
        delta_str = f"value {original_pred['value']} → {new_pred['value']} ({delta:+.4f})"

    overrides_str = ", ".join(f"{k}={v!r}" for k, v in overrides.items())
    readout = (
        f"What-if analysis for row {row_idx} of model {model_id[:8]} "
        f"(target: '{meta.target_col}'). Overrides: {overrides_str}. Result: {delta_str}."
    )

    return {
        "model_id": model_id,
        "target_col": meta.target_col,
        "task_type": meta.task_type,
        "row_idx": row_idx,
        "overrides": overrides,
        "original_prediction": original_pred,
        "new_prediction": new_pred,
        "delta": delta,
        "original_feature_values": {
            k: (None if pd.isna(v) else v)
            for k, v in original_row.iloc[0].to_dict().items()
        },
        "engineering_readout": readout,
    }
