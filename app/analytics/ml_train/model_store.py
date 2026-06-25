from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings

import logging

logger = logging.getLogger(__name__)


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
    optimal_threshold: float | None = None
    lag_config: dict | None = None
    onnx_path: str | None = None
    training_stats: dict | None = None
    conformal_halfwidth: float | None = None
    conformal_classification_threshold: float | None = None
    data_fingerprint: dict | None = None
    is_champion: bool = False
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

    def get_champion(self, dataset_id: str | None, target_col: str) -> "ModelMeta | None":
        """Return the current champion model for (dataset_id, target_col), or None."""
        reg = self._load_registry()
        known = ModelMeta.__dataclass_fields__.keys()
        champions = [
            ModelMeta(**{k: v for k, v in d.items() if k in known})
            for d in reg.get("models", {}).values()
            if d.get("dataset_id") == dataset_id
            and d.get("target_col") == target_col
            and d.get("is_champion", False)
        ]
        return champions[0] if champions else None

    def promote(self, model_id: str) -> None:
        """Mark model_id as champion; demote all others for the same (dataset_id, target_col)."""
        reg = self._load_registry()
        models = reg.get("models", {})
        if model_id not in models:
            raise KeyError(f"Unknown model_id: {model_id}")
        dataset_id = models[model_id].get("dataset_id")
        target_col = models[model_id].get("target_col")
        for mid, mdict in models.items():
            if mdict.get("dataset_id") == dataset_id and mdict.get("target_col") == target_col:
                mdict["is_champion"] = mid == model_id
        self._save_registry(reg)
        logger.info("promoted model %s as champion (dataset_id=%s target=%s)", model_id, dataset_id, target_col)

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
        optimal_threshold: float | None = None,
        lag_config: dict | None = None,
        onnx_path: str | None = None,
        training_stats: dict | None = None,
        conformal_halfwidth: float | None = None,
        conformal_classification_threshold: float | None = None,
        data_fingerprint: dict | None = None,
        model_id: str | None = None,
    ) -> ModelMeta:
        model_id = model_id or str(uuid.uuid4())
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
            optimal_threshold=optimal_threshold,
            lag_config=lag_config,
            onnx_path=onnx_path,
            training_stats=training_stats,
            conformal_halfwidth=conformal_halfwidth,
            conformal_classification_threshold=conformal_classification_threshold,
            data_fingerprint=data_fingerprint,
        )

        reg = self._load_registry()
        reg.setdefault("models", {})
        reg["models"][model_id] = meta.__dict__
        self._save_registry(reg)
        self._evict_oldest(dataset_id, target_col)
        return meta

    def get_meta(self, model_id: str) -> ModelMeta:
        reg = self._load_registry()
        models = reg.get("models", {})
        if model_id not in models:
            raise KeyError(f"Unknown model_id: {model_id}")
        known = ModelMeta.__dataclass_fields__.keys()
        return ModelMeta(**{k: v for k, v in models[model_id].items() if k in known})

    def load_model(self, model_id: str) -> tuple[Any, ModelMeta]:
        meta = self.get_meta(model_id)
        pipeline = joblib.load(meta.path)
        return pipeline, meta

    def list_models(self) -> list[ModelMeta]:
        reg = self._load_registry()
        known = ModelMeta.__dataclass_fields__.keys()
        return [ModelMeta(**{k: v for k, v in d.items() if k in known}) for d in reg.get("models", {}).values()]

    def delete_model(self, model_id: str) -> None:
        reg = self._load_registry()
        models = reg.get("models", {})
        if model_id not in models:
            return
        known = ModelMeta.__dataclass_fields__.keys()
        meta = ModelMeta(**{k: v for k, v in models[model_id].items() if k in known})
        for artifact in (meta.path, meta.onnx_path):
            if artifact:
                p = Path(artifact)
                if p.exists():
                    p.unlink()
        del models[model_id]
        self._save_registry(reg)

    def _evict_oldest(self, dataset_id: str | None, target_col: str) -> None:
        """Remove the oldest models for (dataset_id, target_col) when the cap is exceeded."""
        cap = settings.model_registry_max_per_target
        reg = self._load_registry()
        candidates = [
            ModelMeta(**v)
            for v in reg.get("models", {}).values()
            if v.get("dataset_id") == dataset_id and v.get("target_col") == target_col
        ]
        if len(candidates) <= cap:
            return
        candidates.sort(key=lambda m: m.created_at)
        to_evict = candidates[: len(candidates) - cap]
        for m in to_evict:
            logger.info("evicting model %s (cap=%d for dataset_id=%s target=%s)", m.model_id, cap, dataset_id, target_col)
            self.delete_model(m.model_id)

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
