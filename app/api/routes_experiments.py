from __future__ import annotations

from fastapi import APIRouter, Query

from app.analytics.ml_train.experiment_tracker import get_tracker

router = APIRouter()


@router.get("")
def list_experiments(
    dataset_id: str | None = Query(default=None),
    target_col: str | None = Query(default=None),
    model_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    return get_tracker().list_runs(
        dataset_id=dataset_id,
        target_col=target_col,
        model_type=model_type,
        limit=limit,
    )


@router.get("/{run_id}")
def get_experiment(run_id: str):
    run = get_tracker().get_run(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run
