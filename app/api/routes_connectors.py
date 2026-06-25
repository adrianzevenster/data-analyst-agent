"""Database and URL connectors.

POST /connectors/postgres  — run a SQL query against a PostgreSQL database
POST /connectors/sqlite    — run a SQL query against an uploaded SQLite file
POST /connectors/url       — import CSV / Parquet / JSON from a remote URL
"""
from __future__ import annotations

import io
import logging
import os
import sqlite3
import tempfile
from typing import Literal

import pandas as pd
import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.analytics.dataset_manager import DatasetManager
from app.core.models import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()
dm = DatasetManager()

try:
    import psycopg2
    _PSYCOPG2 = True
except ImportError:
    _PSYCOPG2 = False

_URL_TIMEOUT = 60  # seconds
_MAX_URL_BYTES = 200 * 1024 * 1024  # 200 MB


# ── PostgreSQL ────────────────────────────────────────────────────────────────

class PostgresRequest(BaseModel):
    connection_string: str
    query: str
    dataset_name: str | None = None


@router.post("/connectors/postgres", response_model=UploadResponse)
def connect_postgres(body: PostgresRequest) -> UploadResponse:
    if not _PSYCOPG2:
        raise HTTPException(
            status_code=422,
            detail="psycopg2 is not installed. Add psycopg2-binary to requirements and restart.",
        )
    try:
        conn = psycopg2.connect(body.connection_string)
        df = pd.read_sql(body.query, conn)
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PostgreSQL error: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Query returned no rows.")

    name = body.dataset_name or "postgres_query.csv"
    if not name.endswith(".csv"):
        name += ".csv"
    meta = dm.register_df(df, filename=name, make_active=True)
    return UploadResponse(
        dataset_id=meta.dataset_id, filename=name,
        n_rows=meta.n_rows, n_cols=meta.n_cols,
        notes=[f"PostgreSQL: {meta.n_rows:,} rows × {meta.n_cols} cols"],
    )


# ── SQLite ────────────────────────────────────────────────────────────────────

@router.post("/connectors/sqlite", response_model=UploadResponse)
async def connect_sqlite(
    file: UploadFile = File(...),
    query: str = Form(default="SELECT * FROM sqlite_master LIMIT 100"),
    dataset_name: str = Form(default=""),
) -> UploadResponse:
    content = await file.read()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        try:
            df = pd.read_sql(query, conn)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"SQLite query error: {exc}")
        finally:
            conn.close()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if df.empty:
        raise HTTPException(status_code=422, detail="Query returned no rows.")

    name = dataset_name.strip() or (file.filename or "sqlite_query.csv")
    if not name.endswith(".csv"):
        name += ".csv"
    meta = dm.register_df(df, filename=name, make_active=True)
    return UploadResponse(
        dataset_id=meta.dataset_id, filename=name,
        n_rows=meta.n_rows, n_cols=meta.n_cols,
        notes=[f"SQLite: {meta.n_rows:,} rows × {meta.n_cols} cols"],
    )


# ── URL ───────────────────────────────────────────────────────────────────────

class UrlRequest(BaseModel):
    url: str
    format: Literal["auto", "csv", "parquet", "json"] = "auto"
    dataset_name: str | None = None


def _detect_format(url: str, content_type: str) -> str:
    url_lower = url.lower().split("?")[0]
    if url_lower.endswith(".parquet"):
        return "parquet"
    if url_lower.endswith(".json") or "json" in content_type:
        return "json"
    return "csv"


@router.post("/connectors/url", response_model=UploadResponse)
def connect_url(body: UrlRequest) -> UploadResponse:
    try:
        resp = requests.get(body.url, timeout=_URL_TIMEOUT, stream=True)
        resp.raise_for_status()
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > _MAX_URL_BYTES:
                raise HTTPException(status_code=422, detail="Remote file exceeds 200 MB limit.")
            chunks.append(chunk)
        raw = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {exc}")

    fmt: str = body.format
    if fmt == "auto":
        fmt = _detect_format(body.url, resp.headers.get("content-type", ""))

    try:
        if fmt == "parquet":
            df = pd.read_parquet(io.BytesIO(raw))
        elif fmt == "json":
            df = pd.read_json(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse response as {fmt}: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Remote file contained no rows.")

    raw_name = body.url.rstrip("/").split("/")[-1].split("?")[0] or "remote"
    name = body.dataset_name or raw_name
    if "." not in name:
        name += f".{fmt}"
    meta = dm.register_df(df, filename=name, make_active=True)
    return UploadResponse(
        dataset_id=meta.dataset_id, filename=name,
        n_rows=meta.n_rows, n_cols=meta.n_cols,
        notes=[f"URL ({fmt}): {meta.n_rows:,} rows × {meta.n_cols} cols"],
    )
