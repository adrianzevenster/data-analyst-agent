from __future__ import annotations

import pandas as pd

from app.analytics.profiling import profile_dataset


def test_profile_dataset_flags_high_missingness_and_skew_but_not_continuous_floats_as_ids():
    df = pd.DataFrame(
        {
            "id": range(50),
            "continuous": [float(i) + 0.5 for i in range(50)],  # all-unique floats
            "mostly_missing": [None] * 30 + list(range(20)),
            "skewed": [1] * 45 + [1000] * 5,
        }
    )

    result = profile_dataset(df)

    findings = result["findings"]
    assert any("'id'" in f and "identifier" in f for f in findings)
    # Continuous floats are expected to be all-unique; must not be flagged as an id.
    assert not any("'continuous'" in f and "identifier" in f for f in findings)
    assert any("mostly_missing" in f and "missing" in f for f in findings)
    assert any("skewed" in f for f in findings)
    assert "engineering_readout" in result
    assert result["charts"]


def test_profile_dataset_handles_boolean_and_datetime_columns_without_crashing():
    df = pd.DataFrame(
        {
            "flag": [True, False, True, True],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        }
    )

    result = profile_dataset(df)

    flag_col = next(c for c in result["columns"] if c["name"] == "flag")
    assert flag_col["true_count"] == 3
    assert flag_col["false_count"] == 1

    date_col = next(c for c in result["columns"] if c["name"] == "event_date")
    assert date_col["min"] is not None
    assert date_col["range_days"] > 0


def test_profile_dataset_categorical_top_values():
    df = pd.DataFrame({"region": ["a", "a", "a", "b"]})

    result = profile_dataset(df)

    region_col = next(c for c in result["columns"] if c["name"] == "region")
    assert region_col["top_values"][0]["value"] == "a"
    assert region_col["top_values"][0]["count"] == 3


def test_profile_dataset_no_findings_readout_is_clean():
    df = pd.DataFrame({"a": [1, 2, 1, 2, 1, 2], "b": [4, 5, 4, 5, 4, 5]})

    result = profile_dataset(df)

    assert result["findings"] == []
    assert "No major structural issues" in result["engineering_readout"]
