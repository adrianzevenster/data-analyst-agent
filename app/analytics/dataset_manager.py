from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings
from app.ingestion.normalizers import normalize_dataframe

@dataclass
class DatasetMeta:
    dataset_id: str
    filename: str
    path: str
    n_rows: int
    n_cols: int
    columns: list[str]


class DatasetManager:
    """
    Stores datasets as Parquet in <data_dir>/uploads and tracks metadata in <data_dir>/dataset_registry.json

    Registry format (current):
      {
        "datasets": {
          "<dataset_id>": { ...DatasetMeta dict... },
          ...
        },
        "active_dataset_id": "<dataset_id>" | null
      }

    Backwards compatibility:
      If registry is an old-style dict { "<dataset_id>": {...meta...}, ... },
      it will be migrated in-memory and saved back in the new format.
    """

    def __init__(self, base_dir: str | None = None):
        # base_dir override still supported, but defaults to settings.data_dir
        self.base_dir = Path(base_dir).resolve() if base_dir else settings.data_path
        self.upload_dir: Path = self.base_dir / "uploads"
        self.registry_path: Path = self.base_dir / "dataset_registry.json"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({"datasets": {}, "active_dataset_id": None})

        # Ensure registry is in the expected shape (migrate if required)
        reg = self._load_registry()
        migrated = self._maybe_migrate_registry(reg)
        if migrated is not None:
            self._save_registry(migrated)

    def _normalize_dataset_path(self, p: str) -> str:
        """
        Make registry paths portable across Docker and host environments:
        - /app/data/... → rebase onto actual base_dir (handles host-runs with Docker-written paths)
        - host-absolute .../data/uploads/... → base_dir/uploads/...
        - relative uploads/... → base_dir/uploads/...
        """
        if not p:
            return str(self.base_dir)

        if p.startswith("/app/data/"):
            # Rebase onto actual base_dir so host runs can read Docker-written paths
            tail = p[len("/app/data/"):]  # e.g. "uploads/<id>.parquet"
            return str(self.base_dir / tail)

        marker = "/data/"
        if marker in p:
            tail = p.split(marker, 1)[1]  # "uploads/<id>.parquet"
            return str(self.base_dir / tail)

        if not p.startswith("/"):
            return str(self.base_dir / p)

        return p

    # ------------------------
    # Registry internals
    # ------------------------
    def _load_registry(self) -> dict:
        with self.registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, reg: dict) -> None:
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

    def _maybe_migrate_registry(self, reg: dict) -> dict | None:
        """
        If reg is old-style: {dataset_id: meta, ...}
        migrate to:
          {"datasets": {dataset_id: meta, ...}, "active_dataset_id": None}
        """
        if "datasets" in reg and "active_dataset_id" in reg:
            # already new-style
            if not isinstance(reg["datasets"], dict):
                reg["datasets"] = {}
                return reg
            return None

        # old-style dict: keys look like UUIDs and values look like meta dicts
        # migrate it
        datasets = dict(reg) if isinstance(reg, dict) else {}
        return {"datasets": datasets, "active_dataset_id": None}

    def _get_reg(self) -> dict:
        reg = self._load_registry()
        migrated = self._maybe_migrate_registry(reg)
        if migrated is not None:
            reg = migrated
            self._save_registry(reg)
        reg.setdefault("datasets", {})
        reg.setdefault("active_dataset_id", None)
        return reg

    # ------------------------
    # Active dataset helpers
    # ------------------------
    def set_active_dataset(self, dataset_id: str) -> None:
        reg = self._get_reg()
        if dataset_id not in reg["datasets"]:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        reg["active_dataset_id"] = dataset_id
        self._save_registry(reg)

    def get_active_dataset_id(self) -> str | None:
        reg = self._get_reg()
        return reg.get("active_dataset_id")

    def get_active_meta(self) -> DatasetMeta | None:
        active_id = self.get_active_dataset_id()
        if not active_id:
            return None
        return self.get_meta(active_id)

    # ------------------------
    # Dataset CRUD
    # ------------------------
    def register_df(self, df: pd.DataFrame, filename: str, make_active: bool = True) -> DatasetMeta:
        df = normalize_dataframe(df)
        dataset_id = str(uuid.uuid4())

        out_path = self.upload_dir / f"{dataset_id}.parquet"
        df.to_parquet(out_path, index=False)

        meta = DatasetMeta(
            dataset_id=dataset_id,
            filename=filename,
            path=str(out_path),
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            columns=list(map(str, df.columns.tolist())),
        )

        reg = self._get_reg()
        reg["datasets"][dataset_id] = meta.__dict__
        if make_active:
            reg["active_dataset_id"] = dataset_id
        self._save_registry(reg)
        return meta

    def get_meta(self, dataset_id: str) -> DatasetMeta:
        reg = self._get_reg()
        datasets = reg["datasets"]
        if dataset_id not in datasets:
            raise KeyError(f"Unknown dataset_id: {dataset_id}")
        return DatasetMeta(**datasets[dataset_id])

    def load_df(
            self,
            dataset_id: str,
            columns: Optional[list[str]] = None,
            limit: Optional[int] = None,
    ) -> pd.DataFrame:
        meta = self.get_meta(dataset_id)
        parquet_path = self._normalize_dataset_path(meta.path)
        df = pd.read_parquet(parquet_path, columns=columns)
        if limit is not None:
            return df.head(limit)
        return df

    def list_datasets(self, include_inactive: bool = True) -> list[DatasetMeta]:
        reg = self._get_reg()
        datasets = reg["datasets"]

        if include_inactive:
            return [DatasetMeta(**v) for v in datasets.values()]

        active_id = reg.get("active_dataset_id")
        if not active_id or active_id not in datasets:
            return []
        return [DatasetMeta(**datasets[active_id])]
