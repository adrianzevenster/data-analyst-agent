from __future__ import annotations

import pandas as pd

from app.analytics.quality import data_quality_report, overrepresented_categories, skewed_features


def test_data_quality_report_handles_boolean_columns_without_crashing():
    df = pd.DataFrame({"flag": [True, False, True, True, False]})

    result = data_quality_report(df)

    # Must not raise, and booleans should be summarised as categorical, not numeric.
    flag_col = next(c for c in result["columns"] if c["name"] == "flag")
    assert "top_values" in flag_col
    assert "skewness" not in flag_col


def test_data_quality_report_flags_high_cardinality_and_missingness():
    df = pd.DataFrame(
        {
            "free_text": [f"note-{i}" for i in range(100)],
            "mostly_missing": [None] * 60 + list(range(40)),
        }
    )

    result = data_quality_report(df)

    findings_text = " ".join(result["findings"])
    assert "free_text" in findings_text and "cardinality" in findings_text
    assert "mostly_missing" in findings_text and "missing" in findings_text
    assert "engineering_readout" in result


def test_data_quality_report_embeds_histograms_for_skewed_columns():
    df = pd.DataFrame({"skewed": [1] * 95 + [1000] * 5})

    result = data_quality_report(df)

    assert result["charts"]
    assert result["charts"][0]["type"] == "histogram"
    assert result["charts"][0]["column"] == "skewed"


def test_overrepresented_categories_includes_chart_and_readout():
    df = pd.DataFrame({"region": ["a"] * 9 + ["b"]})

    result = overrepresented_categories(df, col="region", threshold=0.5)

    assert result["dominant_values"]
    assert result["charts"][0]["type"] == "bar"
    assert "a" in result["engineering_readout"]


def test_overrepresented_categories_unknown_column_returns_error():
    df = pd.DataFrame({"region": ["a", "b"]})

    result = overrepresented_categories(df, col="missing_col")

    assert "error" in result


def test_skewed_features_returns_dict_with_charts_and_readout():
    df = pd.DataFrame({"skewed": [1] * 95 + [1000] * 5, "normal": range(100)})

    result = skewed_features(df, threshold=1.0)

    assert result["features"][0]["column"] == "skewed"
    assert result["charts"][0]["type"] == "histogram"
    assert "skewed" in result["engineering_readout"]


def test_skewed_features_excludes_boolean_columns():
    df = pd.DataFrame({"flag": [True, False] * 50})

    result = skewed_features(df, threshold=0.1)

    assert all(item["column"] != "flag" for item in result["features"])


def test_skewed_features_no_matches_has_clean_readout():
    df = pd.DataFrame({"a": range(100)})

    result = skewed_features(df, threshold=10.0)

    assert result["features"] == []
    assert "No numeric features" in result["engineering_readout"]
