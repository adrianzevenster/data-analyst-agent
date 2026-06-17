from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.insights import auto_insights


def test_auto_insights_combines_multiple_analyses():
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "revenue": np.linspace(100, 300, n) + rng.normal(0, 5, n),
            "region": rng.choice(["east", "west"], n),
        }
    )
    df["cost"] = df["revenue"] * 0.8 + rng.normal(0, 2, n)

    result = auto_insights(df)

    assert "data_quality" in result["analyses_run"]
    assert "relationships" in result["analyses_run"]
    assert "trend" in result["analyses_run"]
    assert result["insights"]
    assert result["insights"][0]["rank"] == 1
    assert result["charts"]
    assert "engineering_readout" in result


def test_auto_insights_does_not_flag_datetime_column_as_identifier():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "value": range(n)})

    result = auto_insights(df)

    finding_texts = " ".join(f["finding"] for f in result["insights"])
    assert "'date'" not in finding_texts or "identifier" not in finding_texts

    # also assert no exception path was hit for the quality section
    assert "error" not in result["analyses_run"]


def test_auto_insights_handles_small_dataset_without_crashing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = auto_insights(df)

    assert "engineering_readout" in result
    assert isinstance(result["insights"], list)


def test_auto_insights_clean_readout_when_no_findings():
    df = pd.DataFrame({"a": [1, 2, 1, 2, 1, 2], "b": [3, 4, 3, 4, 3, 4]})

    result = auto_insights(df)

    assert isinstance(result["engineering_readout"], str)
