from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.trends import trend_analysis


def test_trend_analysis_detects_upward_trend():
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": dates, "revenue": np.linspace(100, 300, n) + rng.normal(0, 5, n)})

    result = trend_analysis(df)

    assert result["direction"] == "upward"
    assert result["overall_change_pct"] > 0
    assert result["charts"][0]["type"] == "line"
    assert "date_col" in result and result["date_col"] == "date"


def test_trend_analysis_overall_change_uses_fitted_line_not_noisy_boundary_buckets():
    # resample() can produce a tiny partial first bucket; the raw first value
    # alone would wildly distort a first-vs-last percentage.
    n = 400
    dates = pd.date_range("2023-01-03", periods=n, freq="D")  # deliberately not aligned to week start
    values = np.linspace(100, 300, n)
    df = pd.DataFrame({"date": dates, "revenue": values})

    result = trend_analysis(df)

    # Values roughly triple end-to-end; sum-aggregated weekly buckets should
    # reflect a comparable overall change, not an order-of-magnitude artifact.
    assert 100 < result["overall_change_pct"] < 400


def test_trend_analysis_autodetects_datetime_from_string_column():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D").astype(str)
    df = pd.DataFrame({"day": dates, "value": range(n)})

    result = trend_analysis(df)

    assert result["date_col"] == "day"


def test_trend_analysis_no_datetime_column_returns_error():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = trend_analysis(df)

    assert "error" in result


def test_trend_analysis_too_few_periods_returns_error():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=1), "value": [100.0]})

    result = trend_analysis(df)

    assert "error" in result


def test_trend_analysis_invalid_freq_returns_error_not_raises():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10), "value": range(10)})

    result = trend_analysis(df, freq="not-a-freq")

    assert "error" in result
