from __future__ import annotations

import pandas as pd
import pytest

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager
from app.analytics.tool_validation import validate_tool_args


# ── multidim_pivot column validation in rule planner ─────────────────────────

def _make_planner(tmp_path, df: pd.DataFrame, monkeypatch) -> tuple[Planner, str]:
    monkeypatch.setenv("ENABLE_RAG", "0")
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(df, "data.csv")
    planner = Planner()
    planner.dm = manager
    return planner, meta.dataset_id


_SALES_DF = pd.DataFrame({
    "Region": ["East", "West", "East"],
    "Category": ["A", "B", "A"],
    "Revenue": [100, 200, 150],
})


def test_pivot_resolves_column_case_insensitively(tmp_path, monkeypatch):
    """'breakdown by region' should resolve to 'Region' (capital R)."""
    planner, dataset_id = _make_planner(tmp_path, _SALES_DF, monkeypatch)
    calls, _, _, _, _ = planner.plan("give me a breakdown by region", dataset_id)

    pivot_calls = [c for c in calls if c.name == "multidim_pivot"]
    assert pivot_calls, "Expected a multidim_pivot call"
    assert pivot_calls[0].arguments["index"] == ["Region"]


def test_pivot_drops_unknown_dimension_names(tmp_path, monkeypatch):
    """Parsed tokens that aren't real columns should be silently dropped."""
    planner, dataset_id = _make_planner(tmp_path, _SALES_DF, monkeypatch)
    calls, _, _, _, _ = planner.plan("breakdown by nonexistent_col", dataset_id)

    pivot_calls = [c for c in calls if c.name == "multidim_pivot"]
    assert pivot_calls
    assert "nonexistent_col" not in pivot_calls[0].arguments["index"]


def test_pivot_keeps_valid_and_drops_invalid_dimensions(tmp_path, monkeypatch):
    """When user mixes valid + invalid dims, keep valid ones only."""
    planner, dataset_id = _make_planner(tmp_path, _SALES_DF, monkeypatch)
    calls, _, _, _, _ = planner.plan("breakdown by region and ghost_col", dataset_id)

    pivot_calls = [c for c in calls if c.name == "multidim_pivot"]
    assert pivot_calls
    idx = pivot_calls[0].arguments["index"]
    assert "Region" in idx
    assert "ghost_col" not in idx


def test_pivot_passes_tokens_through_when_no_dataset(monkeypatch):
    """Without a dataset, fall through to the raw parsed tokens (LLM path)."""
    monkeypatch.setenv("ENABLE_RAG", "0")
    planner = Planner()
    planner.dm = DatasetManager()
    calls, _, _, _, _ = planner.plan("breakdown by region", dataset_id=None)

    pivot_calls = [c for c in calls if c.name == "multidim_pivot"]
    assert pivot_calls
    # Without a df the token is passed as-is (lowercase from the message)
    assert "region" in pivot_calls[0].arguments["index"]


def test_pivot_multi_dim_resolves_both_columns(tmp_path, monkeypatch):
    """'breakdown by region and category' resolves both columns."""
    planner, dataset_id = _make_planner(tmp_path, _SALES_DF, monkeypatch)
    calls, _, _, _, _ = planner.plan("breakdown by region and category", dataset_id)

    pivot_calls = [c for c in calls if c.name == "multidim_pivot"]
    assert pivot_calls
    idx = pivot_calls[0].arguments["index"]
    assert "Region" in idx
    assert "Category" in idx


# ── validate_tool_args extensions ────────────────────────────────────────────

_DF = pd.DataFrame({"revenue": [1, 2], "region": ["a", "b"]})


def test_overrepresented_categories_rejects_unknown_col():
    with pytest.raises(ValueError, match="col not in dataset"):
        validate_tool_args(_DF, "overrepresented_categories", {"col": "does_not_exist"})


def test_overrepresented_categories_accepts_valid_col():
    validate_tool_args(_DF, "overrepresented_categories", {"col": "region"})


def test_overrepresented_categories_accepts_no_col():
    validate_tool_args(_DF, "overrepresented_categories", {})


def test_skewed_features_rejects_unknown_cols():
    with pytest.raises(ValueError, match="cols contains unknown columns"):
        validate_tool_args(_DF, "skewed_features", {"cols": ["revenue", "ghost"]})


def test_skewed_features_accepts_valid_cols():
    validate_tool_args(_DF, "skewed_features", {"cols": ["revenue"]})


def test_missingness_matrix_rejects_unknown_cols():
    with pytest.raises(ValueError, match="cols contains unknown columns"):
        validate_tool_args(_DF, "missingness_matrix", {"cols": ["ghost"]})


def test_missingness_matrix_accepts_empty_cols():
    validate_tool_args(_DF, "missingness_matrix", {"cols": []})
