"""Tests for dataset fingerprinting and model lineage tracking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.scoring import score_with_model
from app.analytics.ml_train.drift import compute_fingerprint, compare_fingerprints


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(0, 0.3, n)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


def test_compute_fingerprint_basic():
    df = _reg_df(n=50)
    fp = compute_fingerprint(df, ["x1", "x2"])
    assert fp["n_rows"] == 50
    assert fp["n_cols"] == 2
    assert sorted(fp["columns"]) == ["x1", "x2"]
    assert "column_hash" in fp
    assert "x1" in fp["numeric_means"]


def test_compare_fingerprints_match():
    df = _reg_df(n=100)
    fp = compute_fingerprint(df, ["x1", "x2"])
    report = compare_fingerprints(df, ["x1", "x2"], fp)
    assert report["lineage_ok"] is True
    assert report["columns_removed"] == []
    assert report["columns_added"] == []
    assert report["distribution_shifted"] == []


def test_compare_fingerprints_removed_column():
    df = _reg_df(n=100)
    fp = compute_fingerprint(df, ["x1", "x2"])
    report = compare_fingerprints(df[["x1"]], ["x1"], fp)
    assert report["lineage_ok"] is False
    assert "x2" in report["columns_removed"]


def test_compare_fingerprints_shifted_distribution():
    df_train = _reg_df(n=100)
    fp = compute_fingerprint(df_train, ["x1", "x2"])
    df_score = df_train.copy()
    df_score["x1"] = df_score["x1"] + 100
    report = compare_fingerprints(df_score, ["x1", "x2"], fp)
    assert report["lineage_ok"] is False
    assert "x1" in report["distribution_shifted"]


def test_fingerprint_stored_in_model_meta(mm):
    df = _reg_df(n=100)
    result = train_supervised_model(
        df, target_col="y", tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    meta = mm.get_meta(result["model_id"])
    assert meta.data_fingerprint is not None
    assert meta.data_fingerprint["n_rows"] > 0


def test_lineage_report_in_scoring_result(mm):
    df = _reg_df(n=100)
    result = train_supervised_model(
        df, target_col="y", tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    scored = score_with_model(df, model_id=result["model_id"], model_manager=mm)
    assert "lineage" in scored
    lineage = scored["lineage"]
    assert lineage is not None
    assert lineage["lineage_ok"] is True
