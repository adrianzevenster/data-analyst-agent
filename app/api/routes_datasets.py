from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.analytics.dataset_manager import DatasetManager

router = APIRouter()
dm = DatasetManager()


@router.get("")
def list_datasets():
    datasets = dm.list_datasets(include_inactive=True)
    active_id = dm.get_active_dataset_id()

    return [
        {**d.__dict__, "active": d.dataset_id == active_id}
        for d in datasets
    ]


@router.get("/active")
def get_active_dataset():
    meta = dm.get_active_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="No active dataset")
    return meta.__dict__


@router.get("/{dataset_id}")
def get_dataset_meta(dataset_id: str):
    return dm.get_meta(dataset_id).__dict__


@router.get("/{dataset_id}/sample")
def sample_dataset(dataset_id: str, limit: int = 50):
    df = dm.load_df(dataset_id, limit=limit)
    return {
        "dataset_id": dataset_id,
        "columns": list(map(str, df.columns)),
        "data": df.to_dict(orient="records"),
    }


@router.get("/active/sample")
def sample_active_dataset(limit: int = 50):
    dataset_id = dm.get_active_dataset_id()
    if not dataset_id:
        raise HTTPException(status_code=404, detail="No active dataset")

    df = dm.load_df(dataset_id, limit=limit)
    return {
        "dataset_id": dataset_id,
        "columns": list(map(str, df.columns)),
        "data": df.to_dict(orient="records"),
    }
