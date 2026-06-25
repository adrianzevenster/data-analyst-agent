"""Tests for Tier-4 features: hypothesis testing and report generation."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Hypothesis testing ────────────────────────────────────────────────────────

def _two_group_df(n: int = 100, diff: float = 1.5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "group": ["A"] * n + ["B"] * n,
        "value": np.concatenate([rng.normal(0, 1, n), rng.normal(diff, 1, n)]),
        "score": rng.normal(5, 2, n * 2),
    })


def test_two_sample_t_detects_difference():
    from app.analytics.hypothesis import hypothesis_test
    df = _two_group_df(diff=2.0)
    r = hypothesis_test(df, test_type="two_sample_t", col_a="value", group_col="group")
    assert "error" not in r
    assert r["significant"]
    assert r["p_value"] < 0.05
    assert "cohens_d" in r
    assert abs(r["cohens_d"]) > 0.5


def test_two_sample_t_no_difference():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30,
        "value": np.concatenate([rng.normal(0, 1, 30), rng.normal(0.05, 1, 30)]),
    })
    r = hypothesis_test(df, test_type="two_sample_t", col_a="value", group_col="group")
    assert "error" not in r
    assert not r["significant"]

def test_two_sample_t_col_a_col_b():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"before": rng.normal(5, 1, 50), "after": rng.normal(7, 1, 50)})
    r = hypothesis_test(df, test_type="two_sample_t", col_a="before", col_b="after")
    assert "error" not in r
    assert r["significant"]

def test_one_sample_t():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(5, 1, 100)})
    # Test against μ₀=5 (should not be significant)
    r = hypothesis_test(df, test_type="one_sample_t", col_a="x", popmean=5.0)
    assert "error" not in r
    assert not r["significant"]
    # Test against μ₀=0 (should be very significant)
    r2 = hypothesis_test(df, test_type="one_sample_t", col_a="x", popmean=0.0)
    assert r2["significant"]

def test_paired_t():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(2)
    before = rng.normal(100, 10, 50)
    df = pd.DataFrame({"before": before, "after": before + rng.normal(5, 2, 50)})
    r = hypothesis_test(df, test_type="paired_t", col_a="before", col_b="after")
    assert "error" not in r
    assert r["significant"]
    assert "mean_diff" in r


def test_mannwhitney():
    from app.analytics.hypothesis import hypothesis_test
    df = _two_group_df(diff=2.0)
    r = hypothesis_test(df, test_type="mannwhitney", col_a="value", group_col="group")
    assert "error" not in r
    assert r["significant"]
    assert "rank_biserial_r" in r


def test_chi_squared():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "gender": rng.choice(["M", "F"], 200),
        "outcome": rng.choice(["pass", "fail"], 200, p=[0.7, 0.3]),
    })
    r = hypothesis_test(df, test_type="chi_squared", col_a="gender", col_b="outcome")
    assert "error" not in r
    assert "chi2_statistic" in r
    assert "cramers_v" in r
    assert "p_value" in r


def test_anova():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "value": np.concatenate([rng.normal(0, 1, 30), rng.normal(2, 1, 30), rng.normal(4, 1, 30)]),
    })
    r = hypothesis_test(df, test_type="anova", col_a="value", group_col="group")
    assert "error" not in r
    assert r["significant"]
    assert "f_statistic" in r
    assert "eta_squared" in r
    assert r["n_groups"] == 3


def test_correlation_test():
    from app.analytics.hypothesis import hypothesis_test
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({"x": x, "y": x * 0.8 + rng.normal(0, 0.3, 100)})
    r = hypothesis_test(df, test_type="correlation", col_a="x", col_b="y")
    assert "error" not in r
    assert r["pearson_significant"]
    assert abs(r["pearson_r"]) > 0.6
    assert "spearman_r" in r


def test_power_analysis_required_n():
    from app.analytics.hypothesis import hypothesis_test
    df = pd.DataFrame({"x": [1.0]})  # dummy
    r = hypothesis_test(df, test_type="power_analysis", effect_size=0.5, target_power=0.8)
    assert "error" not in r
    assert "required_n_per_group" in r
    # Cohen's d=0.5, power=0.8 → ~64 per group
    assert 50 <= r["required_n_per_group"] <= 80
    assert "power_curve" in r


def test_power_analysis_achieved_power():
    from app.analytics.hypothesis import hypothesis_test
    df = pd.DataFrame({"x": [1.0]})
    r = hypothesis_test(df, test_type="power_analysis", effect_size=0.8, n_obs=30)
    assert "error" not in r
    assert "achieved_power" in r
    # Large effect (d=0.8) with n=30 should have decent power
    assert r["achieved_power"] > 0.5


def test_hypothesis_missing_col_returns_error():
    from app.analytics.hypothesis import hypothesis_test
    df = pd.DataFrame({"a": [1, 2, 3]})
    r = hypothesis_test(df, test_type="two_sample_t", col_a="nonexistent", group_col="a")
    assert "error" in r


def test_hypothesis_engineering_readout():
    from app.analytics.hypothesis import hypothesis_test
    df = _two_group_df()
    r = hypothesis_test(df, test_type="two_sample_t", col_a="value", group_col="group")
    assert "engineering_readout" in r
    assert len(r["engineering_readout"]) > 10


# ── Report generation ──────────────────────────────────────────────────────────

def test_template_report_from_tool_results():
    from app.api.routes_reports import _template_report
    tool_results = [
        {
            "name": "profile_dataset",
            "ok": True,
            "result": {
                "n_rows": 1000,
                "n_cols": 10,
                "engineering_readout": "Dataset has 1000 rows and 10 columns.",
            },
        },
        {
            "name": "anomaly_scan",
            "ok": True,
            "result": {
                "n_anomalies": 5,
                "anomaly_rate_pct": 0.5,
                "engineering_readout": "5 anomalies detected (0.5%).",
            },
        },
    ]
    report = _template_report(tool_results, dataset_id="test_ds")
    assert "# Data Analysis Report" in report
    assert "Dataset Profile" in report
    assert "Anomaly Detection" in report
    assert "1000 rows" in report or "n_rows" in report.lower() or "Dataset" in report


def test_template_report_empty_results():
    from app.api.routes_reports import _template_report
    report = _template_report([], dataset_id=None)
    assert "# Data Analysis Report" in report


def test_report_endpoint_no_results(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    # Non-existent conversation → no tool results → 422
    resp = client.post("/reports/generate", json={"conversation_id": "nonexistent-conv-id"})
    assert resp.status_code == 422


def test_collect_tool_results_empty_conversation():
    from app.api.routes_reports import _collect_tool_results
    results = _collect_tool_results("totally-nonexistent-conv")
    assert isinstance(results, list)
    assert len(results) == 0
