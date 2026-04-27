from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings


def ensure_data_paths() -> None:
    data_dir: Path = settings.data_path
    uploads_dir: Path = settings.upload_path
    corpus_dir: Path = settings.corpus_path
    indexes_dir: Path = settings.index_path
    registry_path: Path = settings.registry_path

    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    if not registry_path.exists():
        registry_path.write_text(
            json.dumps({"datasets": {}, "active_dataset_id": None}, indent=2),
            encoding="utf-8",
        )
