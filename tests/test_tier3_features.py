"""Tests for Tier-3 features: causal inference and cross-dataset analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Causal inference ───────────────────────────────────────────────────────────

def _causal_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n).astype(float)   # binary
    confounder = rng.normal(0, 1, size=n)
    outcome = 2.0 * treatment + 0.5 * confounder + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "confounder": confounder})


def _continuous_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(5, 2, size=n)
    y = 1.5 * x + rng.normal(0, 1, size=n)
    return pd.DataFrame({"dose": x, "response": y})


def test_causal_effect_basic_binary():
    from app.analytics.causal import estimate_causal_effect
    df = _causal_df()
    result = estimate_causal_effect(df, treatment_col="treatment", outcome_col="outcome")
    assert "error" not in result
    assert result["treatment_is_binary"] is True
    assert result["significant_at_05"] is True
    assert result["effect_direction"] == "positive"
    # True ATE is 2.0 — estimate should be within 0.5
    assert abs(result["ate"] - 2.0) < 0.5


def test_causal_effect_with_confounder():
    from app.analytics.causal import estimate_causal_effect
    df = _causal_df()
    result = estimate_causal_effect(
        df,
        treatment_col="treatment",
        outcome_col="outcome",
        control_cols=["confounder"],
    )
    assert "error" not in result
    assert result["control_cols"] == ["confounder"]
    # Controlling for confounders should give estimate closer to 2.0
    assert abs(result["ate"] - 2.0) < 0.4


def test_causal_effect_continuous_treatment():
    from app.analytics.causal import estimate_causal_effect
    df = _continuous_df()
    result = estimate_causal_effect(df, treatment_col="dose", outcome_col="response")
    assert "error" not in result
    assert result["treatment_is_binary"] is False
    assert result["effect_metric"] == "standardised_beta"
    assert result["significant_at_05"] is True
    # True slope is 1.5
    assert abs(result["ate"] - 1.5) < 0.3


def test_causal_effect_e_value_positive():
    from app.analytics.causal import estimate_causal_effect
    df = _causal_df()
    result = estimate_causal_effect(df, treatment_col="treatment", outcome_col="outcome")
    # E-value must be >= 1; larger effects → larger E-values
    assert result["e_value"] >= 1.0


def test_causal_effect_missing_col_returns_error():
    from app.analytics.causal import estimate_causal_effect
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = estimate_causal_effect(df, treatment_col="x", outcome_col="nonexistent")
    assert "error" in result


def test_causal_effect_too_few_rows_returns_error():
    from app.analytics.causal import estimate_causal_effect
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    result = estimate_causal_effect(df, treatment_col="x", outcome_col="y")
    assert "error" in result


def test_causal_effect_mediation():
    from app.analytics.causal import estimate_causal_effect

    rng = np.random.default_rng(99)
    n = 300
    treatment = rng.integers(0, 2, size=n).astype(float)
    mediator = 0.8 * treatment + rng.normal(0, 0.3, size=n)
    outcome = 0.5 * treatment + 1.2 * mediator + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame({"T": treatment, "M": mediator, "Y": outcome})

    result = estimate_causal_effect(df, treatment_col="T", outcome_col="Y", mediation_col="M")
    assert "error" not in result
    med = result["mediation"]
    assert med is not None
    assert "indirect_effect" in med
    assert "direct_effect" in med
    assert "mediation_pct" in med
    # Indirect path should be substantial (T → M → Y is the dominant path)
    assert med["mediation_pct"] > 30.0


def test_causal_effect_engineering_readout_present():
    from app.analytics.causal import estimate_causal_effect
    df = _causal_df()
    result = estimate_causal_effect(df, treatment_col="treatment", outcome_col="outcome")
    assert "engineering_readout" in result
    assert "ATE" in result["engineering_readout"]
    assert "E-value" in result["engineering_readout"]


# ── Cross-dataset profile ──────────────────────────────────────────────────────

def _make_dm_with_two_datasets(tmp_path):
    """Return (DatasetManager, df_a, dataset_id_a, dataset_id_b)."""
    from app.analytics.dataset_manager import DatasetManager
    import app.core.config as cfg_module

    cfg_module.settings.data_dir = str(tmp_path)
    dm = DatasetManager()

    # Dataset A: customer orders
    df_a = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "region": ["north", "south", "east", "west", "north"],
    })
    # Dataset B: customer demographics
    df_b = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 34, 45, 28, 52],
        "income": [40000, 80000, 60000, 95000, 70000],
    })

    meta_a = dm.register_df(df_a, filename="orders.csv")
    meta_b = dm.register_df(df_b, filename="demographics.csv")
    return dm, df_a, meta_a.dataset_id, meta_b.dataset_id


def test_cross_dataset_discovers_join_key(tmp_path, monkeypatch):
    from app.analytics.cross_dataset import cross_dataset_profile
    import app.core.config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "data_dir", str(tmp_path))
    dm, df_a, id_a, id_b = _make_dm_with_two_datasets(tmp_path)

    result = cross_dataset_profile(df_a, dataset_id_a=id_a)
    assert "error" not in result
    assert result["n_datasets_compared"] >= 1

    # The join key customer_id should be found
    comp = result["comparisons"][0]
    cols = {c["col_a"] for c in comp["join_key_candidates"]}
    assert "customer_id" in cols


def test_cross_dataset_computes_correlations(tmp_path, monkeypatch):
    from app.analytics.cross_dataset import cross_dataset_profile
    import app.core.config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "data_dir", str(tmp_path))
    dm, df_a, id_a, id_b = _make_dm_with_two_datasets(tmp_path)

    result = cross_dataset_profile(df_a, dataset_id_a=id_a)
    comp = result["comparisons"][0]
    # There should be cross-correlations since revenue and income share structure
    assert "cross_correlations" in comp
    assert isinstance(comp["cross_correlations"], list)


def test_cross_dataset_no_other_datasets(tmp_path, monkeypatch):
    from app.analytics.cross_dataset import cross_dataset_profile
    import app.core.config as cfg_module

    # Point to a fresh empty dir — no datasets registered
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(cfg_module.settings, "data_dir", str(empty_dir))

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = cross_dataset_profile(df, dataset_id_a="sole_dataset")
    assert "error" in result


def test_cross_dataset_readout_present(tmp_path, monkeypatch):
    from app.analytics.cross_dataset import cross_dataset_profile
    import app.core.config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "data_dir", str(tmp_path))
    _, df_a, id_a, _ = _make_dm_with_two_datasets(tmp_path)

    result = cross_dataset_profile(df_a, dataset_id_a=id_a)
    assert "engineering_readout" in result
    assert len(result["engineering_readout"]) > 10


def test_name_similarity_utility():
    from app.analytics.cross_dataset import _name_similarity

    assert _name_similarity("customer_id", "customer_id") == 1.0
    assert _name_similarity("cust_id", "customer_id") > 0.0
    assert _name_similarity("revenue", "age") == 0.0
