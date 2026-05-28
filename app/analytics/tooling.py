# app/analytics/tooling.py
from __future__ import annotations

from app.analytics.registry import AnalyticsToolRegistry, Tool

from app.analytics.profiling import profile_dataset
from app.analytics.multidim import multidim_pivot
from app.analytics.sql import duckdb_query
from app.analytics.anomalies import anomaly_scan
from app.analytics.clustering import kmeans_clusters
from app.analytics.viz_specs import simple_bar_spec
from app.analytics.ml_eval import evaluate_ml_predictions

from app.analytics.quality import (
    data_quality_report,
    missingness_matrix,
    overrepresented_categories,
    skewed_features,
)

_registry: AnalyticsToolRegistry | None = None


def get_registry() -> AnalyticsToolRegistry:
    global _registry
    if _registry is not None:
        return _registry

    r = AnalyticsToolRegistry()

    # Existing tools
    r.register(Tool("profile_dataset", "Summarize columns, missingness, basic stats.", profile_dataset))
    r.register(Tool("multidim_pivot", "Create a pivot (multi-dim aggregation).", multidim_pivot))
    r.register(Tool("duckdb_query", "Run SQL over the dataset table 't'.", duckdb_query))
    r.register(Tool("anomaly_scan", "Detect outliers using IsolationForest on numeric columns.", anomaly_scan))
    r.register(Tool("kmeans_clusters", "Cluster rows using KMeans on numeric columns.", kmeans_clusters))
    r.register(Tool("simple_bar_spec", "Generate a simple bar chart spec from x/y columns.", simple_bar_spec))

    # New quality tools
    r.register(Tool("data_quality_report", "Detailed stats incl. missing %, skewness, and percentiles.", data_quality_report))
    r.register(Tool("missingness_matrix", "Columns with highest missing ratios.", missingness_matrix))
    r.register(Tool("overrepresented_categories", "Find dominant values in a categorical column.", overrepresented_categories))
    r.register(Tool("skewed_features", "List numeric features with high skewness.", skewed_features))
    r.register(
        Tool(
            "evaluate_ml_predictions",
            (
                "Evaluate ML prediction outputs using classification, regression, "
                "forecasting, probability-score, and precomputed metric diagnostics."
            ),
            evaluate_ml_predictions,
        )
    )

    _registry = r
    return _registry
