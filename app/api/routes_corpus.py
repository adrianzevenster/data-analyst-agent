from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.rag.corpus_ingest import ingest_corpus

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_SUFFIXES = {".txt", ".md", ".pdf"}


class CorpusFile(BaseModel):
    filename: str
    size_bytes: int
    modified_at: float


class CorpusUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


class CorpusDeleteResponse(BaseModel):
    chunks_indexed: int


class CorpusListResponse(BaseModel):
    files: list[CorpusFile]


def _corpus_path() -> Path:
    p = Path(settings.corpus_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _run_ingest() -> dict:
    """Run the blocking ingest_corpus() in a thread so we don't stall the event loop."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, ingest_corpus)
    except Exception as e:
        logger.exception("ingest_corpus failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@router.get("", response_model=CorpusListResponse)
def list_corpus():
    root = _corpus_path()
    files: list[CorpusFile] = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in _ALLOWED_SUFFIXES:
            stat = f.stat()
            files.append(CorpusFile(
                filename=str(f.relative_to(root)),
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
            ))
    return CorpusListResponse(files=files)


@router.post("/upload", response_model=CorpusUploadResponse)
async def upload_corpus_file(file: UploadFile = File(...)):
    filename = file.filename or "upload.txt"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{suffix}'. Allowed: .txt, .md, .pdf",
        )

    dest = _corpus_path() / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest.write_bytes(content)

    result = await _run_ingest()
    return CorpusUploadResponse(filename=filename, chunks_indexed=result["chunks_indexed"])


@router.delete("/files/{filename:path}", response_model=CorpusDeleteResponse)
async def delete_corpus_file(filename: str):
    target = _corpus_path() / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        target.resolve().relative_to(_corpus_path().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target.unlink()
    result = await _run_ingest()
    return CorpusDeleteResponse(chunks_indexed=result["chunks_indexed"])
