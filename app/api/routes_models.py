from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.analytics.ml_train.model_store import ModelManager

router = APIRouter()
_manager = ModelManager()


class ModelMetaOut(BaseModel):
    model_id: str
    task_type: str
    model_type: str
    target_col: str
    feature_cols: list[str]
    dataset_id: str | None = None
    created_at: str


@router.get("", response_model=list[ModelMetaOut])
def list_models():
    return [ModelMetaOut(**m.__dict__) for m in _manager.list_models()]


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: str):
    manager = ModelManager()
    try:
        meta = manager.get_meta(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    try:
        os.remove(meta.path)
    except FileNotFoundError:
        pass
    reg = manager._load_registry()
    reg["models"].pop(model_id, None)
    manager._save_registry(reg)
