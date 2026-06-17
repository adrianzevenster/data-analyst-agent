from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.relationships import correlation_analysis


def test_correlation_analysis_finds_strong_numeric_pair():
    rng = np.random.default_rng(0)
    n = 200
    revenue = rng.normal(100, 20, n)
    df = pd.DataFrame({"revenue": revenue, "cost": revenue * 0.8 + rng.normal(0, 1, n)})

    result = correlation_analysis(df)

    assert result["numeric_correlations"][0]["column_a"] in {"revenue", "cost"}
    assert result["numeric_correlations"][0]["abs_correlation"] > 0.9
    assert any("strongly" in f for f in result["findings"])
    assert result["charts"][0]["type"] == "scatter"


def test_correlation_analysis_finds_categorical_association():
    df = pd.DataFrame(
        {
            "region": ["high"] * 50 + ["low"] * 50,
            "revenue": [200.0] * 50 + [50.0] * 50,
        }
    )

    result = correlation_analysis(df)

    assert result["categorical_associations"][0]["categorical_col"] == "region"
    assert result["categorical_associations"][0]["correlation_ratio"] > 0.9


def test_correlation_analysis_suppresses_sparse_overlap_findings():
    n = 200
    df = pd.DataFrame(
        {
            "a": list(range(n)),
            # only the last 10% of rows are non-null and happen to align with "a"
            "sparse": [None] * (n - 20) + list(range(20)),
        }
    )

    result = correlation_analysis(df)

    sparse_pair = next(
        p for p in result["numeric_correlations"]
        if "sparse" in (p["column_a"], p["column_b"])
    )
    assert sparse_pair["low_overlap"] is True
    assert not any("sparse" in f for f in result["findings"])


def test_correlation_analysis_handles_insufficient_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = correlation_analysis(df)

    assert result["numeric_correlations"] == []
    assert result["categorical_associations"] == []
    assert "Not enough" in result["engineering_readout"]
