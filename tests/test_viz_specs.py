from __future__ import annotations

import pandas as pd

from app.analytics.viz_specs import (
    histogram_spec,
    line_spec,
    multi_series_bar_spec,
    scatter_spec,
    simple_bar_spec,
)


def test_simple_bar_spec_includes_axis_labels():
    df = pd.DataFrame({"region": ["a", "b"], "revenue": [10, 20]})

    spec = simple_bar_spec(df, x="region", y="revenue")

    assert spec["type"] == "bar"
    assert spec["x_label"] == "region"
    assert spec["y_label"] == "revenue"
    assert len(spec["data"]) == 2


def test_multi_series_bar_spec_keeps_all_value_columns():
    df = pd.DataFrame({"region": ["a", "b"], "revenue": [10, 20], "cost": [5, 8]})

    spec = multi_series_bar_spec(df, x="region", y_cols=["revenue", "cost"])

    assert spec["y_series"] == ["revenue", "cost"]
    assert spec["data"][0]["revenue"] == 10
    assert spec["data"][0]["cost"] == 5


def test_histogram_spec_produces_ordered_bins_covering_full_range():
    df = pd.DataFrame({"x": list(range(100))})

    spec = histogram_spec(df, column="x", bins=10)

    assert spec["type"] == "histogram"
    assert len(spec["data"]) == 10
    assert spec["data"][0]["bin_start"] <= 0
    assert spec["data"][-1]["bin_end"] >= 99
    assert sum(b["count"] for b in spec["data"]) == 100


def test_histogram_spec_handles_empty_column_gracefully():
    df = pd.DataFrame({"x": [None, None]})

    spec = histogram_spec(df, column="x")

    assert spec["data"] == []


def test_line_spec_sorts_by_x():
    df = pd.DataFrame({"day": [3, 1, 2], "value": [30, 10, 20]})

    spec = line_spec(df, x="day", y="value")

    assert [row["value"] for row in spec["data"]] == [10, 20, 30]


def test_scatter_spec_computes_correlation_for_numeric_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

    spec = scatter_spec(df, x="x", y="y")

    assert spec["correlation"] == 1.0


def test_scatter_spec_skips_correlation_for_non_numeric_columns():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})

    spec = scatter_spec(df, x="x", y="y")

    assert spec["correlation"] is None
