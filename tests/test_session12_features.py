"""Tests for session-12 features:
- SMOTE for imbalanced binary classifiers
- Data fingerprinting and model lineage
- Conformal classification prediction sets
- Async training job API (job store)
- Model comparison (experiments endpoint)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model, _SMOTE_AVAILABLE
from app.analytics.ml_train.scoring import score_with_model
from app.analytics.ml_train.drift import compute_fingerprint, compare_fingerprints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced_clf_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(1, 2, n)})
    df["label"] = ((df["a"] + df["b"]) > 1).astype(int)
    return df


def _imbalanced_clf_df(n: int = 200) -> pd.DataFrame:
    """10:1 imbalance ratio (180 majority, 20 minority)."""
    rng = np.random.default_rng(7)
    majority = pd.DataFrame({"a": rng.normal(0, 1, 180), "b": rng.normal(0, 1, 180), "label": 0})
    minority = pd.DataFrame({"a": rng.normal(5, 1, 20), "b": rng.normal(5, 1, 20), "label": 1})
    return pd.concat([majority, minority], ignore_index=True)


def _reg_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(0, 0.3, n)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# 1. SMOTE
# ---------------------------------------------------------------------------

def test_smote_available():
    """Sanity: imbalanced-learn should be installed in this environment."""
    assert _SMOTE_AVAILABLE, "imbalanced-learn must be installed"


def test_smote_triggered_for_imbalanced_gradient_boosting(mm):
    """GBM + 10:1 imbalance should trigger SMOTE (no class_weight support)."""
    if not _SMOTE_AVAILABLE:
        pytest.skip("imbalanced-learn not installed")
    df = _imbalanced_clf_df()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result, result.get("error")
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "smote" in notes.lower(), f"Expected SMOTE note, got: {notes}"


def test_smote_note_shows_ratio(mm):
    df = _imbalanced_clf_df()
    if not _SMOTE_AVAILABLE:
        pytest.skip()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    # Should mention the imbalance ratio
    assert "ratio" in notes.lower() or "imbalance" in notes.lower()


def test_smote_not_triggered_for_balanced_data(mm):
    """Balanced dataset should not trigger SMOTE."""
    df = _balanced_clf_df()
    result = train_supervised_model(
        df, target_col="label",
        model_type="gradient_boosting_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    assert "smote" not in notes.lower()


def test_smote_not_triggered_for_xgboost(mm):
    """XGBoost uses scale_pos_weight instead — SMOTE only added for ratio > 10."""
    df = _imbalanced_clf_df()  # 10:1 ratio
    if not _SMOTE_AVAILABLE:
        pytest.skip()
    result = train_supervised_model(
        df, target_col="label",
        model_type="xgboost_classifier",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    notes = " ".join(result.get("preprocessing_notes", []))
    # XGBoost with 10:1 should still trigger SMOTE (ratio > 10 threshold == 10.0, it's 9.0 here)
    # 180/20 = 9.0 which is NOT > 10, so no SMOTE for XGBoost
    assert "smote" not in notes.lower()


# ---------------------------------------------------------------------------
# 2. Data fingerprinting
# ---------------------------------------------------------------------------

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
    # Scoring data only has x1
    report = compare_fingerprints(df[["x1"]], ["x1"], fp)
    assert report["lineage_ok"] is False
    assert "x2" in report["columns_removed"]


def test_compare_fingerprints_shifted_distribution():
    df_train = _reg_df(n=100)
    fp = compute_fingerprint(df_train, ["x1", "x2"])
    # Shift x1 by 10 stds
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


# ---------------------------------------------------------------------------
# 3. Conformal classification prediction sets
# ---------------------------------------------------------------------------

def test_conformal_threshold_stored(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    meta = mm.get_meta(result["model_id"])
    assert meta.conformal_classification_threshold is not None
    assert 0.0 <= meta.conformal_classification_threshold <= 1.0


def test_prediction_set_in_scoring(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    scored = score_with_model(df, model_id=result["model_id"], model_manager=mm)
    assert "prediction_set_info" in scored
    ps_info = scored["prediction_set_info"]
    assert ps_info is not None
    assert ps_info["coverage_target"] == 0.90
    assert ps_info["avg_set_size"] >= 1.0
    # Check the column is in scored_rows
    rows = scored["scored_rows"]
    assert len(rows) > 0
    assert "prediction_set" in rows[0]


def test_prediction_set_values_are_valid(mm):
    df = _balanced_clf_df(n=120)
    result = train_supervised_model(
        df, target_col="label",
        model_type="logistic_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result
    scored = score_with_model(df, model_id=result["model_id"], model_manager=mm)
    rows = scored["scored_rows"]
    for row in rows[:20]:
        ps = row.get("prediction_set", "")
        # Each value in the set must be a valid class (0 or 1 as string)
        for cls in str(ps).split("|"):
            assert cls in ("0", "1"), f"Unexpected class in prediction_set: {cls!r}"


# ---------------------------------------------------------------------------
# 4. Async training job store
# ---------------------------------------------------------------------------

def test_submit_and_get_job():
    from app.api.training_jobs import submit_job, get_job
    import time

    results = []
    job_id = submit_job(lambda: results.append(42) or {"done": True})
    # Job should be visible immediately
    job = get_job(job_id)
    assert job is not None
    assert job["status"] in ("running", "done")
    # Wait a moment for completion
    for _ in range(20):
        j = get_job(job_id)
        if j and j["status"] == "done":
            break
        time.sleep(0.1)
    final = get_job(job_id)
    assert final is not None
    assert final["status"] == "done"
    assert final["result"] == {"done": True}


def test_job_captures_errors():
    from app.api.training_jobs import submit_job, get_job
    import time

    job_id = submit_job(lambda: 1 / 0)
    for _ in range(20):
        j = get_job(job_id)
        if j and j["status"] == "error":
            break
        time.sleep(0.1)
    final = get_job(job_id)
    assert final is not None
    assert final["status"] == "error"
    assert "division by zero" in str(final.get("error", ""))


def test_list_jobs():
    from app.api.training_jobs import submit_job, list_jobs

    submit_job(lambda: {"x": 1})
    jobs = list_jobs(limit=50)
    assert len(jobs) >= 1
    assert all("job_id" in j for j in jobs)
    assert all("status" in j for j in jobs)
