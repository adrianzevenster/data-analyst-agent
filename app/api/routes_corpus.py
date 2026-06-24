from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.rag.corpus_ingest import ingest_corpus

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
    total_chunks: int | None = None


def _corpus_path() -> Path:
    p = Path(settings.corpus_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("", response_model=CorpusListResponse)
def list_corpus():
    """Return all files currently in the corpus directory."""
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
    """Save a document to the corpus and rebuild the RAG index."""
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

    result = ingest_corpus()
    return CorpusUploadResponse(filename=filename, chunks_indexed=result["chunks_indexed"])


@router.delete("/files/{filename:path}", response_model=CorpusDeleteResponse)
def delete_corpus_file(filename: str):
    """Remove a document from the corpus and rebuild the RAG index."""
    target = _corpus_path() / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Guard against path traversal
    try:
        target.resolve().relative_to(_corpus_path().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target.unlink()
    result = ingest_corpus()
    return CorpusDeleteResponse(chunks_indexed=result["chunks_indexed"])
