"""Dataset annotations — per-dataset business context and column notes indexed into RAG."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from app.core.config import settings
from app.rag.corpus_ingest import ingest_corpus

logger = logging.getLogger(__name__)
router = APIRouter()

_ingest_lock = threading.Lock()


def _annotations_dir() -> Path:
    p = settings.data_path / "annotations"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _annotation_file(dataset_id: str) -> Path:
    return _annotations_dir() / f"{dataset_id}.json"


def _corpus_annotation_path(dataset_id: str) -> Path:
    corpus = Path(settings.corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    return corpus / f"_dataset_{dataset_id}_annotations.md"


def _load(dataset_id: str) -> "AnnotationData":
    path = _annotation_file(dataset_id)
    if not path.exists():
        return AnnotationData()
    try:
        return AnnotationData(**json.loads(path.read_text("utf-8")))
    except Exception:
        return AnnotationData()


def _write_corpus_file(dataset_id: str, ann: "AnnotationData", filename: str) -> None:
    lines = [f"# Dataset: {filename or dataset_id}", ""]
    if ann.description:
        lines += ["## Business Context", ann.description, ""]
    if ann.columns:
        lines += ["## Column Descriptions", ""]
        for col, note in ann.columns.items():
            if note.strip():
                lines.append(f"- **{col}**: {note}")
        lines.append("")
    _corpus_annotation_path(dataset_id).write_text("\n".join(lines), encoding="utf-8")


def _remove_corpus_file(dataset_id: str) -> None:
    p = _corpus_annotation_path(dataset_id)
    if p.exists():
        p.unlink()


def _trigger_reindex() -> None:
    with _ingest_lock:
        try:
            ingest_corpus()
        except Exception as exc:
            logger.warning("Annotation reindex failed: %s", exc)


class AnnotationData(BaseModel):
    description: str = ""
    columns: dict[str, str] = {}


class AnnotationResponse(BaseModel):
    dataset_id: str
    description: str
    columns: dict[str, str]


class SaveAnnotationsRequest(BaseModel):
    description: str = ""
    columns: dict[str, str] = {}
    dataset_filename: str = ""


@router.get("/{dataset_id}", response_model=AnnotationResponse)
def get_annotations(dataset_id: str) -> AnnotationResponse:
    ann = _load(dataset_id)
    return AnnotationResponse(dataset_id=dataset_id, description=ann.description, columns=ann.columns)


@router.put("/{dataset_id}", response_model=AnnotationResponse)
def save_annotations(
    dataset_id: str,
    req: SaveAnnotationsRequest,
    background_tasks: BackgroundTasks,
) -> AnnotationResponse:
    ann = AnnotationData(
        description=req.description.strip(),
        columns={k: v for k, v in req.columns.items() if v.strip()},
    )
    _annotation_file(dataset_id).write_text(ann.model_dump_json(), encoding="utf-8")

    if ann.description or ann.columns:
        _write_corpus_file(dataset_id, ann, req.dataset_filename)
    else:
        _remove_corpus_file(dataset_id)

    background_tasks.add_task(_trigger_reindex)
    return AnnotationResponse(dataset_id=dataset_id, description=ann.description, columns=ann.columns)


@router.delete("/{dataset_id}")
def clear_annotations(dataset_id: str, background_tasks: BackgroundTasks) -> dict:
    _annotation_file(dataset_id).unlink(missing_ok=True)
    _remove_corpus_file(dataset_id)
    background_tasks.add_task(_trigger_reindex)
    return {"status": "cleared"}
