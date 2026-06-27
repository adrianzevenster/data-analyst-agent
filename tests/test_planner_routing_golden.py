"""Planner routing golden set — tool-selection regression gate.

Each case asserts that the rule-based planner routes a specific message to the
correct tool. Unlike test_rule_planner_accuracy_floor.py (which only checks the
aggregate accuracy floor), this test asserts 90%+ correctness on a golden set
where the expected tool is unambiguous. Individual failures surface the exact
mis-routing, making regressions easy to diagnose.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager

ROUTING_FLOOR = 0.90

GENERIC_DF = pd.DataFrame({
    "region": ["east", "west", "east", "west", "north", "south"] * 5,
    "revenue": [100.0, 200.0, 150.0, 50.0, 300.0, 80.0] * 5,
    "units": [1, 2, 3, 4, 5, 1] * 5,
    "cost": [40.0, 80.0, 60.0, 20.0, 120.0, 30.0] * 5,
})

TIME_DF = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=24, freq="MS"),
    "revenue": list(range(100, 124)),
    "units": list(range(10, 34)),
})

CHURN_DF = pd.DataFrame({
    "customer_id": range(1, 21),
    "actual": ([1, 0] * 10),
    "churn_prediction": ([1, 1, 0, 0] * 5),
    "probability": [0.9, 0.6, 0.2, 0.1] * 5,
    "tenure_months": list(range(1, 21)),
    "monthly_spend": [50.0 + i * 10 for i in range(20)],
})

SALES_DF = pd.DataFrame({
    "region": ["east", "west", "north", "south"] * 5,
    "product": ["A", "B", "C", "D"] * 5,
    "channel": ["online", "store"] * 10,
    "revenue": [float(i * 50) for i in range(1, 21)],
    "units": list(range(1, 21)),
})

ML_DF = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60] * 3,
    "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000] * 3,
    "churn": [0, 0, 0, 1, 1, 1, 0, 1] * 3,
})

CAUSAL_DF = pd.DataFrame({
    "treatment": [0, 1, 0, 1, 0, 1, 0, 1] * 3,
    "outcome": [10.0, 20.0, 12.0, 22.0, 9.0, 19.0, 11.0, 21.0] * 3,
    "age": [30, 35, 40, 45, 50, 55, 60, 65] * 3,
    "income": [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000] * 3,
})

# (message, df, expected_tool_name)
ROUTING_CASES = [
    # --- profiling ---
    ("Give me a full profile of this dataset", GENERIC_DF, "profile_dataset"),
    ("Profile this data", GENERIC_DF, "profile_dataset"),
    ("Summarize the dataset", GENERIC_DF, "profile_dataset"),
    ("Show me descriptive statistics", GENERIC_DF, "profile_dataset"),

    # --- data quality ---
    ("Run a data quality healthcheck", GENERIC_DF, "data_quality_report"),
    ("Check data quality", GENERIC_DF, "data_quality_report"),
    ("Are there any missing values?", GENERIC_DF, "missingness_matrix"),
    ("Find duplicate rows", GENERIC_DF, "data_quality_report"),

    # --- anomaly detection ---
    ("Find outliers in this data", GENERIC_DF, "anomaly_scan"),
    ("Detect anomalies", GENERIC_DF, "anomaly_scan"),
    ("Are there any anomalies?", GENERIC_DF, "anomaly_scan"),
    ("Which rows are outliers?", GENERIC_DF, "anomaly_scan"),

    # --- clustering ---
    ("Cluster these rows into groups", GENERIC_DF, "kmeans_clusters"),
    ("Segment customers into groups", SALES_DF, "kmeans_clusters"),
    ("Run k-means clustering", GENERIC_DF, "kmeans_clusters"),
    ("Find natural groupings in the data", GENERIC_DF, "kmeans_clusters"),

    # --- correlation ---
    ("What correlations exist?", GENERIC_DF, "correlation_analysis"),
    ("Which features are correlated?", GENERIC_DF, "correlation_analysis"),
    ("Show me the correlation matrix", GENERIC_DF, "correlation_analysis"),
    ("Are revenue and units correlated?", GENERIC_DF, "correlation_analysis"),

    # --- trend analysis ---
    ("Show me the trend over time", TIME_DF, "trend_analysis"),
    ("How has revenue changed over time?", TIME_DF, "trend_analysis"),
    ("Plot the time series", TIME_DF, "trend_analysis"),
    ("What is the growth trend?", TIME_DF, "trend_analysis"),

    # --- multidim pivot ---
    ("Break down revenue by region", SALES_DF, "multidim_pivot"),
    ("Breakdown of revenue by region and product", SALES_DF, "multidim_pivot"),
    ("Pivot revenue by channel", SALES_DF, "multidim_pivot"),

    # --- SQL (prefix must be lowercase "sql:" per planner convention) ---
    ("sql: SELECT region, SUM(revenue) FROM t GROUP BY region", SALES_DF, "duckdb_query"),
    ("sql: SELECT * FROM t WHERE revenue > 100", SALES_DF, "duckdb_query"),
    ("SQL: SELECT COUNT(*) FROM t", GENERIC_DF, "duckdb_query"),

    # --- ML evaluation ---
    ("How accurate are the churn predictions?", CHURN_DF, "evaluate_ml_predictions"),
    ("What is the F1 score?", CHURN_DF, "evaluate_ml_predictions"),
    ("Show me the confusion matrix", CHURN_DF, "evaluate_ml_predictions"),
    ("Calculate precision and recall", CHURN_DF, "evaluate_ml_predictions"),

    # --- training ---
    ("Train a model to predict revenue", GENERIC_DF, "train_supervised_model"),
    ("Train a random forest to predict churn", ML_DF, "train_supervised_model"),
    ("Build an XGBoost model to predict income", ML_DF, "train_supervised_model"),
    ("Fit a classifier to predict churn", ML_DF, "train_supervised_model"),
    ("Build a regression model for revenue", GENERIC_DF, "train_supervised_model"),

    # --- causal inference ---
    ("What is the causal effect of treatment on outcome", CAUSAL_DF, "estimate_causal_effect"),
    ("Does treatment cause a change in outcome", CAUSAL_DF, "estimate_causal_effect"),
    ("Estimate the treatment effect of treatment on outcome", CAUSAL_DF, "estimate_causal_effect"),
    ("Run a causal analysis on treatment and outcome", CAUSAL_DF, "estimate_causal_effect"),

    # --- anomaly explanation ---
    ("Why is row 5 an anomaly?", GENERIC_DF, "explain_anomaly"),
    ("Explain why row 3 was flagged as an outlier", GENERIC_DF, "explain_anomaly"),
    ("What makes row 10 anomalous?", GENERIC_DF, "explain_anomaly"),

    # --- cross-dataset ---
    ("Compare these datasets", GENERIC_DF, "cross_dataset_profile"),
    ("Find relationships across datasets", GENERIC_DF, "cross_dataset_profile"),
    ("Cross dataset analysis", GENERIC_DF, "cross_dataset_profile"),

]


@pytest.fixture
def rule_planner(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    monkeypatch.setenv("LLM_ENABLED", "false")
    manager = DatasetManager(base_dir=str(tmp_path))
    p = Planner()
    p.dm = manager
    return p, manager


def test_planner_routing_golden(rule_planner):
    p, manager = rule_planner
    hits = 0
    failures = []

    for message, df, expected_tool in ROUTING_CASES:
        dataset_id = manager.register_df(df, "dataset.csv").dataset_id
        calls, _, source, _, _ = p.plan(message, dataset_id)
        selected = {c.name for c in calls}
        hit = expected_tool in selected
        if hit:
            hits += 1
        else:
            failures.append({
                "message": message,
                "expected": expected_tool,
                "selected": sorted(selected),
            })

    accuracy = hits / len(ROUTING_CASES)
    assert accuracy >= ROUTING_FLOOR, (
        f"Routing accuracy {accuracy:.2f} fell below floor {ROUTING_FLOOR}. "
        f"Misses ({len(failures)}):\n"
        + "\n".join(
            f"  {f['message']!r}: expected {f['expected']!r}, got {f['selected']}"
            for f in failures
        )
    )
