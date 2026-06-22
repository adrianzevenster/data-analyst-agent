"""Background training job API.

Provides fire-and-forget model training via a thread-pool backed job registry.
POST /training/jobs  →  submits training, returns job_id immediately (HTTP 202)
GET  /training/jobs  →  list recent jobs (status only)
GET  /training/jobs/{job_id}  →  full job status + result/error when done
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.analytics.dataset_manager import DatasetManager
from app.analytics.ml_train import train_supervised_model
from app.analytics.ml_train.model_store import ModelManager
from app.api.training_jobs import get_job, list_jobs, submit_job

logger = logging.getLogger(__name__)
router = APIRouter()

_dm = DatasetManager()
_mm = ModelManager()


class TrainingJobRequest(BaseModel):
    dataset_id: str
    target_col: str
    model_type: str = "auto"
    tune: bool = True
    cv_folds: int = Field(default=5, ge=2, le=10)


@router.post("/jobs", status_code=202)
def start_training_job(req: TrainingJobRequest) -> dict:
    """Submit a training job. Returns job_id immediately; training runs in background."""
    try:
        df = _dm.load_df(req.dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id!r} not found")

    job_id = submit_job(
        train_supervised_model,
        df,
        req.target_col,
        model_type=req.model_type,
        tune=req.tune,
        cv_folds=req.cv_folds,
        dataset_id=req.dataset_id,
        model_manager=_mm,
    )
    logger.info("Submitted background training job %s (dataset=%s target=%s)", job_id, req.dataset_id, req.target_col)
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs")
def list_training_jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    """List recent training jobs (status summary, no result payload)."""
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "created_at": j["created_at"],
            "completed_at": j.get("completed_at"),
        }
        for j in list_jobs(limit)
    ]


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str) -> dict:
    """Get full status and result for a specific training job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id!r} not found")
    return {"job_id": job_id, **job}
