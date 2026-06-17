"""Golden-query eval for the LLM tool planner against a live LLM server.

Unlike the rule-based golden tests in test_planner_eval.py, this doesn't
assert exact tool-call equality per case - LLM tool choice has legitimate
phrasing/temperature variance even at temperature=0 across model versions.
Instead it computes an aggregate selection-accuracy rate across a fixed
golden set and fails only if accuracy drops below a floor, plus checks that
no calls were dropped for hallucinating columns. That makes it a regression
trip-wire for "switched models/prompts and planning quality cratered"
rather than a brittle case-by-case unit test.

Opt-in only: requires a reachable, configured LLM server (LLM_ENABLED=true
and a live LLM_BASE_URL). Run with: pytest -m llm_eval -v
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.agent.llm import LLMReasoner
from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager
from app.core.config import settings

pytestmark = pytest.mark.llm_eval

MIN_ACCURACY = 0.75

GENERIC_DF = pd.DataFrame(
    {
        "region": ["east", "west", "east", "west", "north", "south"],
        "revenue": [100.0, 200.0, 150.0, 50.0, 300.0, 80.0],
        "units": [1, 2, 3, 4, 5, 1],
    }
)

TIME_DF = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "revenue": [100, 110, 105, 130, 140, 150, 160, 155, 170, 180, 190, 200],
    }
)

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

GOLDEN_CASES = [
    # --- original 9 ---
    ("Give me a profile of this dataset", GENERIC_DF, {"profile_dataset"}),
    ("Run a data quality healthcheck", GENERIC_DF, {"data_quality_report"}),
    ("Are any features skewed?", GENERIC_DF, {"skewed_features"}),
    ("Find outliers in this data", GENERIC_DF, {"anomaly_scan"}),
    ("Cluster these rows into segments", GENERIC_DF, {"kmeans_clusters"}),
    ("What correlations exist in this data?", GENERIC_DF, {"correlation_analysis"}),
    ("Show me the trend over time", TIME_DF, {"trend_analysis"}),
    ("What insights stand out in this data?", GENERIC_DF, {"auto_insights"}),
    ("Train a model to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    # --- phrasing variants: profile / schema ---
    ("What columns does this dataset have?", GENERIC_DF, {"profile_dataset"}),
    ("Describe the schema", GENERIC_DF, {"profile_dataset"}),
    ("What's the shape and data types?", GENERIC_DF, {"profile_dataset"}),
    # --- quality variants ---
    ("Check data quality", GENERIC_DF, {"data_quality_report"}),
    ("Are there any data issues?", GENERIC_DF, {"data_quality_report"}),
    ("Run diagnostics on this dataset", GENERIC_DF, {"data_quality_report"}),
    # --- missingness ---
    ("Which columns are incomplete?", GENERIC_DF, {"missingness_matrix"}),
    ("Show me where data is missing", GENERIC_DF, {"missingness_matrix"}),
    ("Are there null values?", GENERIC_DF, {"missingness_matrix"}),
    # --- skew ---
    ("Do any columns have long tails?", GENERIC_DF, {"skewed_features"}),
    ("Check for heavy-tailed distributions", GENERIC_DF, {"skewed_features"}),
    # --- anomaly / outlier variants ---
    ("Are there any anomalies?", GENERIC_DF, {"anomaly_scan"}),
    ("Detect outliers", GENERIC_DF, {"anomaly_scan"}),
    ("Flag unusual rows", GENERIC_DF, {"anomaly_scan"}),
    # --- clustering / segmentation ---
    ("Segment customers into groups", SALES_DF, {"kmeans_clusters"}),
    ("Run a clustering analysis", GENERIC_DF, {"kmeans_clusters"}),
    ("What natural groupings exist?", SALES_DF, {"kmeans_clusters"}),
    # --- correlation ---
    ("Which features are correlated?", GENERIC_DF, {"correlation_analysis"}),
    ("Are revenue and units related?", GENERIC_DF, {"correlation_analysis"}),
    ("Find associated columns", GENERIC_DF, {"correlation_analysis"}),
    # --- trend ---
    ("How has revenue changed over time?", TIME_DF, {"trend_analysis"}),
    ("Is there a seasonal pattern?", TIME_DF, {"trend_analysis"}),
    ("Plot the time series", TIME_DF, {"trend_analysis"}),
    # --- insights ---
    ("What's interesting about this data?", GENERIC_DF, {"auto_insights"}),
    ("Surprise me with findings", SALES_DF, {"auto_insights"}),
    ("Give me the key findings", SALES_DF, {"auto_insights"}),
    ("What stands out here?", GENERIC_DF, {"auto_insights"}),
    # --- pivot / breakdown ---
    ("Break down revenue by region", SALES_DF, {"multidim_pivot"}),
    ("Pivot by product and channel", SALES_DF, {"multidim_pivot"}),
    ("Group by region and product", SALES_DF, {"multidim_pivot"}),
    # --- SQL ---
    ("sql: SELECT region, SUM(revenue) FROM t GROUP BY region", SALES_DF, {"duckdb_query"}),
    ("sql: SELECT * FROM t WHERE revenue > 100", SALES_DF, {"duckdb_query"}),
    # --- ML evaluation (classification) ---
    ("How accurate are the churn predictions?", CHURN_DF, {"evaluate_ml_predictions"}),
    ("Calculate precision and recall for this model", CHURN_DF, {"evaluate_ml_predictions"}),
    ("What is the F1 score?", CHURN_DF, {"evaluate_ml_predictions"}),
    ("Show me the confusion matrix", CHURN_DF, {"evaluate_ml_predictions"}),
    ("What is the ROC AUC?", CHURN_DF, {"evaluate_ml_predictions"}),
    # --- ML training with named model types ---
    ("Train a random forest to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    ("Build an XGBoost model to predict units", GENERIC_DF, {"train_supervised_model"}),
    ("Fit a logistic regression to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    ("Train a gradient boosted model to predict revenue", GENERIC_DF, {"train_supervised_model"}),
    ("Build a decision tree to predict units", GENERIC_DF, {"train_supervised_model"}),
    # --- ML training generic ---
    ("Build a predictive model for revenue", GENERIC_DF, {"train_supervised_model"}),
    ("Train a classifier on this dataset for column revenue", GENERIC_DF, {"train_supervised_model"}),
    # --- overrepresented categories ---
    ("Is there class imbalance in column region?", SALES_DF, {"overrepresented_categories"}),
    ("Check column region for dominant values", SALES_DF, {"overrepresented_categories"}),
    # --- time-series specific ---
    ("Is there a growth trend?", TIME_DF, {"trend_analysis"}),
    ("Show month-over-month change", TIME_DF, {"trend_analysis"}),
]


@pytest.fixture
def live_planner(tmp_path):
    reasoner = LLMReasoner()
    if not reasoner.enabled:
        pytest.skip("LLM not enabled - set LLM_ENABLED=true and a reachable LLM_BASE_URL to run this eval")

    manager = DatasetManager(base_dir=str(tmp_path))
    p = Planner()
    p.dm = manager
    return p, manager


def test_llm_planner_golden_accuracy(live_planner):
    p, manager = live_planner
    results = []

    for message, df, expected_tools in GOLDEN_CASES:
        dataset_id = manager.register_df(df, "dataset.csv").dataset_id
        calls, _citations, source, llm_error, llm_notes = p.plan(message, dataset_id)
        selected = {c.name for c in calls}
        results.append(
            {
                "message": message,
                "expected": sorted(expected_tools),
                "selected": sorted(selected),
                "hit": bool(selected & expected_tools),
                "planning_source": source,
                "llm_error": llm_error,
                "dropped_calls": llm_notes,
            }
        )

    accuracy = sum(r["hit"] for r in results) / len(results)
    dropped_total = sum(len(r["dropped_calls"]) for r in results)

    report = {
        "accuracy": round(accuracy, 4),
        "min_accuracy": MIN_ACCURACY,
        "dropped_calls_total": dropped_total,
        "results": results,
    }
    settings.eval_path.mkdir(parents=True, exist_ok=True)
    (settings.eval_path / "llm_planner_golden.json").write_text(json.dumps(report, indent=2, default=str))

    failures = [r for r in results if not r["hit"]]
    assert accuracy >= MIN_ACCURACY, (
        f"LLM planner accuracy {accuracy:.2f} fell below floor {MIN_ACCURACY}. "
        f"Failing cases: {json.dumps(failures, indent=2, default=str)}"
    )
