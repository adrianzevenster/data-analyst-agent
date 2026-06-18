from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings


@dataclass
class ModelMeta:
    model_id: str
    path: str
    task_type: str
    model_type: str
    target_col: str
    feature_cols: list[str]
    dataset_id: str | None = None
    log_transform_target: bool = False
    evaluation: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelManager:
    """
    Persists trained sklearn pipelines as joblib files under <data_dir>/models
    and tracks metadata in <data_dir>/model_registry.json, mirroring how
    DatasetManager stores datasets and their registry.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else settings.data_path
        self.model_dir = self.base_dir / "models"
        self.registry_path = self.base_dir / "model_registry.json"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({"models": {}})

    def _load_registry(self) -> dict:
        with self.registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, reg: dict) -> None:
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

    def save_model(
        self,
        pipeline: Any,
        *,
        task_type: str,
        model_type: str,
        target_col: str,
        feature_cols: list[str],
        dataset_id: str | None = None,
        log_transform_target: bool = False,
        evaluation: dict | None = None,
    ) -> ModelMeta:
        model_id = str(uuid.uuid4())
        path = self.model_dir / f"{model_id}.joblib"
        joblib.dump(pipeline, path)

        meta = ModelMeta(
            model_id=model_id,
            path=str(path),
            task_type=task_type,
            model_type=model_type,
            target_col=target_col,
            feature_cols=list(feature_cols),
            dataset_id=dataset_id,
            log_transform_target=log_transform_target,
            evaluation=evaluation or {},
        )

        reg = self._load_registry()
        reg.setdefault("models", {})
        reg["models"][model_id] = meta.__dict__
        self._save_registry(reg)
        return meta

    def get_meta(self, model_id: str) -> ModelMeta:
        reg = self._load_registry()
        models = reg.get("models", {})
        if model_id not in models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return ModelMeta(**models[model_id])

    def load_model(self, model_id: str) -> tuple[Any, ModelMeta]:
        meta = self.get_meta(model_id)
        pipeline = joblib.load(meta.path)
        return pipeline, meta

    def list_models(self) -> list[ModelMeta]:
        reg = self._load_registry()
        return [ModelMeta(**v) for v in reg.get("models", {}).values()]

    def find_previous(
        self, dataset_id: str | None, target_col: str
    ) -> ModelMeta | None:
        """Return the most recently trained model for the same (dataset_id, target_col), or None."""
        reg = self._load_registry()
        candidates = [
            ModelMeta(**v)
            for v in reg.get("models", {}).values()
            if v.get("dataset_id") == dataset_id and v.get("target_col") == target_col
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.created_at)
