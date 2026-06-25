from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.scoring import score_with_model

router = APIRouter()
_manager = ModelManager()


class ModelMetaOut(BaseModel):
    model_id: str
    task_type: str
    model_type: str
    target_col: str
    feature_cols: list[str]
    dataset_id: str | None = None
    log_transform_target: bool = False
    lag_config: dict | None = None
    onnx_path: str | None = None
    is_champion: bool = False
    created_at: str


@router.get("", response_model=list[ModelMetaOut])
def list_models():
    return [ModelMetaOut(**m.__dict__) for m in _manager.list_models()]


@router.get("/{model_id}/download")
def download_model(model_id: str):
    try:
        meta = _manager.get_meta(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    path = Path(meta.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    filename = f"{meta.model_type}__{meta.target_col}__{model_id[:8]}.joblib"
    return FileResponse(path=str(path), media_type="application/octet-stream", filename=filename)


@router.get("/{model_id}/download-onnx")
def download_model_onnx(model_id: str):
    try:
        meta = _manager.get_meta(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    if not meta.onnx_path:
        raise HTTPException(
            status_code=404,
            detail="No ONNX artifact for this model. ONNX export is only available for pipelines without custom transformers.",
        )
    path = Path(meta.onnx_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="ONNX file not found on disk")
    filename = f"{meta.model_type}__{meta.target_col}__{model_id[:8]}.onnx"
    return FileResponse(path=str(path), media_type="application/octet-stream", filename=filename)


@router.post("/{model_id}/score-file")
async def score_model_file(model_id: str, file: UploadFile = File(...)):
    try:
        _manager.get_meta(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    try:
        result = score_with_model(df, model_id, model_manager=_manager)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    out_df = pd.DataFrame(result["scored_rows"])
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    filename = f"predictions__{model_id[:8]}.csv"
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
