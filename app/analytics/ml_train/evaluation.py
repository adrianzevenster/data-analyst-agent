from __future__ import annotations

from app.analytics.ml_train.model_store import ModelManager


def evaluate_trained_model(
    df,
    model_id: str,
    model_manager: ModelManager | None = None,
) -> dict:
    """Return the persisted holdout evaluation for a stored trained model."""
    manager = model_manager or ModelManager()
    try:
        meta = manager.get_meta(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model metadata: {exc}"}

    if not meta.evaluation:
        return {
            "model_id": model_id,
            "task_type": meta.task_type,
            "model_type": meta.model_type,
            "target_col": meta.target_col,
            "error": "No persisted evaluation is available for this model.",
        }

    missing_features = [col for col in meta.feature_cols if col not in df.columns]
    dataset_note = (
        "Current dataset contains all model features."
        if not missing_features
        else f"Current dataset is missing model feature(s): {', '.join(missing_features)}."
    )
    if meta.dataset_id is not None:
        dataset_note += f" Model was trained from dataset {meta.dataset_id}."

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "feature_cols": meta.feature_cols,
        "evaluation": meta.evaluation,
        "optimal_threshold": meta.optimal_threshold,
        "log_transform_target": meta.log_transform_target,
        "conformal_halfwidth": meta.conformal_halfwidth,
        "dataset_note": dataset_note,
        "engineering_readout": (
            f"Loaded persisted holdout evaluation for model {model_id} "
            f"({meta.model_type}, {meta.task_type}) predicting '{meta.target_col}'. "
            f"{dataset_note}"
        ),
    }
