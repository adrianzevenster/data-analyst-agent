"""Rule-based planner accuracy floor — CI regression gate.

Runs a representative subset of the LLM golden cases through the rule-based
planner with the LLM disabled. Asserts that the rule-based baseline stays
above a floor; individual misses are acceptable as long as the LLM covers
them. This test runs in every CI build (no live LLM needed).
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager

RULE_BASELINE_MIN_ACCURACY = 0.55

GENERIC_DF = pd.DataFrame({
    "region": ["east", "west", "east", "west", "north", "south"],
    "revenue": [100.0, 200.0, 150.0, 50.0, 300.0, 80.0],
    "units": [1, 2, 3, 4, 5, 1],
})

TIME_DF = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
    "revenue": [100, 110, 105, 130, 140, 150, 160, 155, 170, 180, 190, 200],
})

CHURN_DF = pd.DataFrame({
    "customer_id": range(1, 8),
    "actual": [1, 0, 1, 0, 1, 0, 1],
    "churn_prediction": [1, 1, 1, 0, 0, 0, 1],
    "probability": [0.9, 0.6, 0.8, 0.2, 0.4, 0.1, 0.7],
    "tenure_months": [3, 24, 6, 48, 12, 36, 2],
    "monthly_spend": [50.0, 120.0, 45.0, 200.0, 75.0, 180.0, 40.0],
})

SALES_DF = pd.DataFrame({
    "region": ["east", "west", "east", "north", "south", "west", "north"],
    "product": ["A", "A", "B", "B", "A", "C", "C"],
    "channel": ["online", "store", "online", "online", "store", "store", "online"],
    "revenue": [200.0, 150.0, 300.0, 100.0, 250.0, 80.0, 120.0],
    "units": [4, 3, 6, 2, 5, 1, 3],
    "returns": [0, 1, 0, 0, 2, 0, 1],
})

# (message, df, expected_tools_any_of) — hit if the selected set intersects expected
BASELINE_CASES = [
    # Core tools
    ("Give me a profile of this dataset", GENERIC_DF, {"profile_dataset"}),
    ("Run a data quality healthcheck", GENERIC_DF, {"data_quality_report"}),
    ("Are any features skewed?", GENERIC_DF, {"skewed_features"}),
    ("Find outliers in this data", GENERIC_DF, {"anomaly_scan"}),
    ("Cluster these rows into segments", GENERIC_DF, {"kmeans_clusters"}),
    ("What correlations exist in this data?", GENERIC_DF, {"correlation_analysis"}),
    ("Show me the trend over time", TIME_DF, {"trend_analysis"}),
    ("What insights stand out in this data?", GENERIC_DF, {"auto_insights"}),
    ("Train a model to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    # Variants
    ("Detect outliers", GENERIC_DF, {"anomaly_scan"}),
    ("Segment customers into groups", SALES_DF, {"kmeans_clusters"}),
    ("Which features are correlated?", GENERIC_DF, {"correlation_analysis"}),
    ("How has revenue changed over time?", TIME_DF, {"trend_analysis"}),
    ("What's interesting about this data?", GENERIC_DF, {"auto_insights"}),
    ("Break down revenue by region", SALES_DF, {"multidim_pivot"}),
    ("sql: SELECT region, SUM(revenue) FROM t GROUP BY region", SALES_DF, {"duckdb_query"}),
    ("How accurate are the churn predictions?", CHURN_DF, {"evaluate_ml_predictions"}),
    ("What is the F1 score?", CHURN_DF, {"evaluate_ml_predictions"}),
    ("Train a random forest to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    ("Build an XGBoost model to predict units", GENERIC_DF, {"train_supervised_model"}),
]


@pytest.fixture
def rule_planner(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    monkeypatch.setenv("LLM_ENABLED", "false")
    manager = DatasetManager(base_dir=str(tmp_path))
    p = Planner()
    p.dm = manager
    return p, manager


def test_rule_planner_accuracy_floor(rule_planner):
    p, manager = rule_planner
    hits = 0
    failures = []

    for message, df, expected_tools in BASELINE_CASES:
        dataset_id = manager.register_df(df, "dataset.csv").dataset_id
        calls, _, source, _, _ = p.plan(message, dataset_id)
        selected = {c.name for c in calls}
        hit = bool(selected & expected_tools)
        if hit:
            hits += 1
        else:
            failures.append({
                "message": message,
                "expected": sorted(expected_tools),
                "selected": sorted(selected),
            })

    accuracy = hits / len(BASELINE_CASES)
    assert accuracy >= RULE_BASELINE_MIN_ACCURACY, (
        f"Rule-based planner accuracy {accuracy:.2f} fell below floor {RULE_BASELINE_MIN_ACCURACY}. "
        f"Failing cases ({len(failures)}):\n"
        + "\n".join(
            f"  {f['message']!r}: expected {f['expected']}, got {f['selected']}"
            for f in failures
        )
    )
