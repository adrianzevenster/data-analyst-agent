from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException

from app.ingestion.loaders import (
    load_csv,
    load_excel,
    load_pdf_text,
    load_image_ocr,
)
from app.analytics.dataset_manager import DatasetManager
from app.analytics.eda_cache import run_and_cache
from app.core.models import UploadResponse

import pandas as pd

router = APIRouter()
dm = DatasetManager()


def _schedule_eda(background_tasks: BackgroundTasks, dataset_id: str, df: pd.DataFrame) -> None:
    """Queue auto-EDA only for tabular datasets (>1 column, numeric content)."""
    if df.shape[1] > 1:
        background_tasks.add_task(run_and_cache, dataset_id, df)


@router.post("", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Upload a file and register it as a dataset.
    The newly uploaded dataset is automatically set as ACTIVE.
    """
    content = await file.read()
    filename = file.filename or "upload"

    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "csv":
        ing = load_csv(content, filename)
        meta = dm.register_df(ing.payload, filename=filename, make_active=True)
        _schedule_eda(background_tasks, meta.dataset_id, ing.payload)
        return UploadResponse(
            dataset_id=meta.dataset_id,
            filename=filename,
            n_rows=meta.n_rows,
            n_cols=meta.n_cols,
            notes=ing.notes,
        )

    if suffix in ("xlsx", "xls"):
        ing = load_excel(content, filename)
        meta = dm.register_df(ing.payload, filename=filename, make_active=True)
        _schedule_eda(background_tasks, meta.dataset_id, ing.payload)
        return UploadResponse(
            dataset_id=meta.dataset_id,
            filename=filename,
            n_rows=meta.n_rows,
            n_cols=meta.n_cols,
            notes=ing.notes,
        )

    if suffix == "parquet":
        import io
        try:
            df = pd.read_parquet(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not read Parquet file: {exc}")
        meta = dm.register_df(df, filename=filename, make_active=True)
        _schedule_eda(background_tasks, meta.dataset_id, df)
        return UploadResponse(
            dataset_id=meta.dataset_id, filename=filename,
            n_rows=meta.n_rows, n_cols=meta.n_cols,
            notes=[f"Loaded Parquet: {meta.n_rows:,} rows × {meta.n_cols} cols"],
        )

    if suffix == "json":
        import io
        try:
            df = pd.read_json(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not read JSON file: {exc}")
        meta = dm.register_df(df, filename=filename, make_active=True)
        _schedule_eda(background_tasks, meta.dataset_id, df)
        return UploadResponse(
            dataset_id=meta.dataset_id, filename=filename,
            n_rows=meta.n_rows, n_cols=meta.n_cols,
            notes=[f"Loaded JSON: {meta.n_rows:,} rows × {meta.n_cols} cols"],
        )

    if suffix == "pdf":
        ing = load_pdf_text(content, filename)
        if not ing.payload:
            raise HTTPException(
                status_code=422,
                detail="No text extracted from PDF. If scanned, try image OCR.",
            )

        df = pd.DataFrame({"text": [ing.payload]})
        meta = dm.register_df(df, filename=filename, make_active=True)
        return UploadResponse(
            dataset_id=meta.dataset_id,
            filename=filename,
            n_rows=meta.n_rows,
            n_cols=meta.n_cols,
            notes=ing.notes,
        )

    if suffix in ("png", "jpg", "jpeg", "webp", "tiff", "bmp"):
        ing = load_image_ocr(content, filename)
        if not ing.payload:
            raise HTTPException(status_code=422, detail="No text extracted from image.")

        df = pd.DataFrame({"text": [ing.payload]})
        meta = dm.register_df(df, filename=filename, make_active=True)
        return UploadResponse(
            dataset_id=meta.dataset_id,
            filename=filename,
            n_rows=meta.n_rows,
            n_cols=meta.n_cols,
            notes=ing.notes,
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {filename}. Supported: CSV, XLSX, XLS, Parquet, JSON, PDF, image.",
    )
