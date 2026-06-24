from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.rag.corpus_ingest import ingest_corpus

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_SUFFIXES = {".txt", ".md", ".pdf"}

# ── ingest state (shared across requests, protected by _ingest_lock) ──────────

_ingest_lock = threading.Lock()
_ingest_state: dict = {"running": False, "chunks_indexed": None, "error": None}


def _do_ingest() -> None:
    """Blocking ingest run; called from a FastAPI BackgroundTask (thread pool)."""
    global _ingest_state
    with _ingest_lock:
        _ingest_state = {"running": True, "chunks_indexed": None, "error": None}
    try:
        result = ingest_corpus()
        with _ingest_lock:
            _ingest_state = {
                "running": False,
                "chunks_indexed": result.get("chunks_indexed"),
                "error": None,
            }
        logger.info("ingest complete: %s chunks", result.get("chunks_indexed"))
    except Exception as exc:
        logger.exception("ingest_corpus failed: %s", exc)
        with _ingest_lock:
            _ingest_state = {"running": False, "chunks_indexed": None, "error": str(exc)}


# ── models ────────────────────────────────────────────────────────────────────


class CorpusFile(BaseModel):
    filename: str
    size_bytes: int
    modified_at: float


class CorpusListResponse(BaseModel):
    files: list[CorpusFile]
    ingest_running: bool = False
    last_chunks_indexed: int | None = None
    last_ingest_error: str | None = None


class CorpusUploadResponse(BaseModel):
    filename: str
    status: str = "indexing"


class CorpusDeleteResponse(BaseModel):
    status: str = "indexing"


# ── helpers ───────────────────────────────────────────────────────────────────


def _corpus_path() -> Path:
    p = Path(settings.corpus_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── routes ────────────────────────────────────────────────────────────────────


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
    with _ingest_lock:
        state = dict(_ingest_state)
    return CorpusListResponse(
        files=files,
        ingest_running=state["running"],
        last_chunks_indexed=state["chunks_indexed"],
        last_ingest_error=state["error"],
    )


@router.post("/upload", response_model=CorpusUploadResponse)
async def upload_corpus_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
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

    background_tasks.add_task(_do_ingest)
    return CorpusUploadResponse(filename=filename, status="indexing")


@router.delete("/files/{filename:path}", response_model=CorpusDeleteResponse)
async def delete_corpus_file(
    filename: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    target = _corpus_path() / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        target.resolve().relative_to(_corpus_path().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target.unlink()
    background_tasks.add_task(_do_ingest)
    return CorpusDeleteResponse(status="indexing")
