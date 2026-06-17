from __future__ import annotations

import pandas as pd

from app.analytics.dataset_manager import DatasetManager


def test_dataset_manager_uses_base_dir(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}), "sample.csv")

    assert meta.path.startswith(str(tmp_path))
    assert (tmp_path / "dataset_registry.json").exists()
    assert (tmp_path / "uploads" / f"{meta.dataset_id}.parquet").exists()

    loaded = manager.load_df(meta.dataset_id)
    assert loaded.to_dict(orient="list") == {"x": [1, 2], "y": ["a", "b"]}


def test_dataset_manager_load_limit(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"x": [1, 2, 3]}), "sample.csv")

    loaded = manager.load_df(meta.dataset_id, limit=2)

    assert loaded["x"].tolist() == [1, 2]
