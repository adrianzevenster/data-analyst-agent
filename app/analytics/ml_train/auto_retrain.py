"""Automated retraining triggered by high feature drift at scoring time.

Called by training_jobs.submit_job when score_with_model detects PSI >= 0.25.
Flow: load dataset → retrain → compare metrics → promote winner.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def auto_retrain_model(
    triggered_by_model_id: str,
    dataset_id: str,
    target_col: str,
    model_type: str = "auto",
) -> dict:
    """Background worker: retrain and promote the winner.

    Returns a result dict consumed by training_jobs and surfaced via GET /jobs/{id}.
    """
    from app.analytics.dataset_manager import DatasetManager
    from app.analytics.ml_train.model_store import ModelManager
    from app.analytics.ml_train.training import train_supervised_model

    dm = DatasetManager()
    manager = ModelManager()

    # Load the full dataset for retraining.
    try:
        df = dm.load_df(dataset_id)
    except Exception as exc:
        return {"status": "failed", "error": f"Could not load dataset {dataset_id}: {exc}"}

    logger.info(
        "auto_retrain: starting retrain for dataset=%s target=%s model_type=%s (triggered by %s)",
        dataset_id, target_col, model_type, triggered_by_model_id,
    )

    try:
        result = train_supervised_model(
            df,
            target_col=target_col,
            model_type=model_type,  # type: ignore[arg-type]
            dataset_id=dataset_id,
            model_manager=manager,
        )
    except Exception as exc:
        return {"status": "failed", "error": f"Training failed: {exc}"}

    if "error" in result:
        return {"status": "failed", "error": result["error"]}

    new_model_id = result.get("model_id")
    comparison = result.get("model_comparison")

    # Promote the winner: new model if improved, keep old otherwise.
    promoted_id: str | None = None
    if comparison:
        if comparison.get("improved"):
            if new_model_id:
                manager.promote(new_model_id)
            promoted_id = new_model_id
            outcome = "new_model_promoted"
        else:
            try:
                manager.promote(triggered_by_model_id)
                promoted_id = triggered_by_model_id
            except KeyError:
                pass
            outcome = "previous_model_retained"
    else:
        # No previous model to compare against — promote new one by default.
        if new_model_id:
            manager.promote(new_model_id)
        promoted_id = new_model_id
        outcome = "new_model_promoted_no_comparison"

    logger.info(
        "auto_retrain: done dataset=%s target=%s outcome=%s promoted=%s",
        dataset_id, target_col, outcome, promoted_id,
    )

    return {
        "status": "done",
        "triggered_by_model_id": triggered_by_model_id,
        "new_model_id": new_model_id,
        "promoted_model_id": promoted_id,
        "outcome": outcome,
        "model_comparison": comparison,
        "new_model_type": result.get("model_type"),
        "new_model_evaluation": result.get("evaluation"),
    }
