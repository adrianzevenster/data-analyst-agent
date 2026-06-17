from __future__ import annotations

import numpy as np
import pandas as pd

from app.agent.llm import LLMReasoner


def test_dataset_analysis_context_includes_categorical_associations_and_outliers():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "region": ["high"] * 100 + ["low"] * 100,
            "revenue": [200.0] * 100 + [50.0] * 100,
        }
    )
    df.loc[0, "revenue"] = 100_000.0  # extreme outlier

    ctx = LLMReasoner.dataset_analysis_context(df)

    assert ctx["strongest_categorical_associations"]
    assert ctx["strongest_categorical_associations"][0]["categorical_column"] == "region"

    revenue_profile = next(c for c in ctx["column_profiles"] if c["name"] == "revenue")
    assert revenue_profile["outlier_count_zscore_3"] >= 1


def test_dataset_analysis_context_does_not_crash_on_boolean_column():
    df = pd.DataFrame({"flag": [True, False, True, True, False]})

    ctx = LLMReasoner.dataset_analysis_context(df)

    flag_profile = next(c for c in ctx["column_profiles"] if c["name"] == "flag")
    assert "top_values" in flag_profile
    assert "numeric_summary" not in flag_profile


def test_dataset_analysis_context_includes_trend_summary_when_datetime_present():
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "revenue": np.linspace(100, 300, n)})

    ctx = LLMReasoner.dataset_analysis_context(df)

    assert ctx["trend_summary"] is not None
    assert ctx["trend_summary"]["direction"] == "upward"


def test_dataset_analysis_context_trend_summary_none_without_datetime():
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx = LLMReasoner.dataset_analysis_context(df)

    assert ctx["trend_summary"] is None


def test_dataset_analysis_context_returns_none_for_no_dataframe():
    assert LLMReasoner.dataset_analysis_context(None) is None
