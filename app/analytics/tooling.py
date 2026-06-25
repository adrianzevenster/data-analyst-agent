# app/analytics/tooling.py
from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.analytics.registry import AnalyticsToolRegistry, Tool, ToolArgs

from app.analytics.profiling import profile_dataset
from app.analytics.multidim import multidim_pivot
from app.analytics.sql import duckdb_query
from app.analytics.anomalies import anomaly_scan, explain_anomaly
from app.analytics.clustering import kmeans_clusters
from app.analytics.viz_specs import simple_bar_spec, histogram_spec, line_spec, scatter_spec
from app.analytics.ml_eval import evaluate_ml_predictions
from app.analytics.ml_train import (
    train_supervised_model,
    score_with_model,
    explain_model,
    shap_explain_prediction,
    evaluate_trained_model,
    forecast_with_model,
    compute_pdp,
)
from app.analytics.ml_train.training import ModelType as TrainingModelType, TaskHint as TrainingTaskHint
from app.analytics.relationships import correlation_analysis
from app.analytics.trends import trend_analysis
from app.analytics.insights import auto_insights
from app.analytics.causal import estimate_causal_effect
from app.analytics.cross_dataset import cross_dataset_profile
from app.analytics.hypothesis import hypothesis_test

from app.analytics.quality import (
    data_quality_report,
    missingness_matrix,
    overrepresented_categories,
    skewed_features,
)

_registry: AnalyticsToolRegistry | None = None


class ProfileDatasetArgs(ToolArgs):
    sample: int = Field(default=5000, ge=1, le=1_000_000)


class MultiDimPivotArgs(ToolArgs):
    index: list[str] = Field(default_factory=list)
    columns: list[str] | None = None
    values: str = ""
    agg: Literal["sum", "mean", "median", "min", "max", "count"] = "sum"
    fillna: float | int | None = 0
    top_n: int = Field(default=50, ge=1, le=10_000)


class DuckDbQueryArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=20_000)


class AnomalyScanArgs(ToolArgs):
    numeric_cols: list[str] = Field(default_factory=list)
    contamination: float = Field(default=0.02, gt=0, lt=0.5)


class ExplainAnomalyArgs(ToolArgs):
    numeric_cols: list[str] = Field(default_factory=list)
    row_idx: int = Field(default=0, ge=0)
    top_k: int = Field(default=10, ge=1, le=50)


class KMeansClustersArgs(ToolArgs):
    numeric_cols: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=2, le=100)


class DataQualityReportArgs(ToolArgs):
    sample: int = Field(default=10000, ge=1, le=1_000_000)


class MissingnessMatrixArgs(ToolArgs):
    top_n: int = Field(default=20, ge=1, le=10_000)


class OverrepresentedCategoriesArgs(ToolArgs):
    col: str = Field(min_length=1)
    threshold: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=10, ge=1, le=10_000)


class SkewedFeaturesArgs(ToolArgs):
    sample: int = Field(default=10000, ge=1, le=1_000_000)
    threshold: float = Field(default=1.0, ge=0)
    max_features: int = Field(default=50, ge=1, le=10_000)


class EvaluateMlPredictionsArgs(ToolArgs):
    actual_col: str | None = None
    prediction_col: str | None = None
    probability_col: str | None = None
    id_col: str | None = None
    task_hint: Literal["auto", "classification", "regression", "forecast", "scored_predictions"] = "auto"
    slice_cols: list[str] | None = None
    top_n: int = Field(default=25, ge=1, le=10_000)


class TrainSupervisedModelArgs(ToolArgs):
    target_col: str = Field(min_length=1)
    feature_cols: list[str] | None = None
    task_hint: TrainingTaskHint = "auto"
    model_type: TrainingModelType = "auto"
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    tune: bool = False
    cv_folds: int = Field(default=2, ge=0, le=10)
    max_rows: int | None = Field(default=10_000, ge=100, le=1_000_000)


class ScoreWithModelArgs(ToolArgs):
    model_id: str = Field(min_length=1)
    top_n: int = Field(default=500, ge=1, le=10_000)


class ExplainModelArgs(ToolArgs):
    model_id: str = Field(min_length=1)
    sample: int = Field(default=500, ge=10, le=10_000)
    n_repeats: int = Field(default=10, ge=3, le=50)


class EvaluateTrainedModelArgs(ToolArgs):
    model_id: str = Field(min_length=1)


class ForecastWithModelArgs(ToolArgs):
    model_id: str = Field(min_length=1)
    horizon: int = Field(default=30, ge=1, le=365)


class ExplainPredictionArgs(ToolArgs):
    model_id: str = Field(min_length=1)
    row_idx: int = Field(default=0, ge=0)


class ComputePdpArgs(ToolArgs):
    model_id: str = Field(min_length=1)
    feature_cols: list[str] | None = None
    n_top_features: int = Field(default=5, ge=1, le=10)
    grid_resolution: int = Field(default=20, ge=5, le=50)


class SimpleBarSpecArgs(ToolArgs):
    x: str = Field(min_length=1)
    y: str = Field(min_length=1)
    title: str = ""


class HistogramSpecArgs(ToolArgs):
    column: str = Field(min_length=1)
    bins: int = Field(default=20, ge=2, le=200)
    title: str = ""


class LineSpecArgs(ToolArgs):
    x: str = Field(min_length=1)
    y: str = Field(min_length=1)
    title: str = ""


class ScatterSpecArgs(ToolArgs):
    x: str = Field(min_length=1)
    y: str = Field(min_length=1)
    title: str = ""


class CorrelationAnalysisArgs(ToolArgs):
    numeric_cols: list[str] | None = None
    categorical_cols: list[str] | None = None
    top_n: int = Field(default=20, ge=1, le=200)


class TrendAnalysisArgs(ToolArgs):
    date_col: str | None = None
    value_col: str | None = None
    freq: Literal["auto", "D", "W", "M", "Q", "Y"] = "auto"
    agg: Literal["sum", "mean", "median", "min", "max", "count"] = "sum"


class AutoInsightsArgs(ToolArgs):
    top_n: int = Field(default=10, ge=1, le=50)


class CausalEffectArgs(ToolArgs):
    treatment_col: str = Field(min_length=1)
    outcome_col: str = Field(min_length=1)
    control_cols: list[str] | None = None
    mediation_col: str | None = None


class CrossDatasetProfileArgs(ToolArgs):
    dataset_id_b: str | None = None


class HypothesisTestArgs(ToolArgs):
    test_type: Literal[
        "two_sample_t", "one_sample_t", "paired_t",
        "mannwhitney", "chi_squared", "anova",
        "correlation", "power_analysis",
    ] = "two_sample_t"
    col_a: str | None = None
    col_b: str | None = None
    group_col: str | None = None
    group_a: str | None = None
    group_b: str | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)
    popmean: float = 0.0
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    effect_size: float | None = None
    n_obs: int | None = None
    target_power: float = Field(default=0.8, gt=0, lt=1)


def get_registry() -> AnalyticsToolRegistry:
    global _registry
    if _registry is not None:
        return _registry

    r = AnalyticsToolRegistry()

    # Existing tools
    r.register(Tool("profile_dataset", "Summarize columns, missingness, basic stats.", profile_dataset, ProfileDatasetArgs))
    r.register(Tool("multidim_pivot", "Create a pivot (multi-dim aggregation).", multidim_pivot, MultiDimPivotArgs))
    r.register(Tool("duckdb_query", "Run SQL over the dataset table 't'.", duckdb_query, DuckDbQueryArgs))
    r.register(Tool("anomaly_scan", "Detect outliers using IsolationForest on numeric columns.", anomaly_scan, AnomalyScanArgs))
    r.register(
        Tool(
            "explain_anomaly",
            (
                "Explain why a specific row is anomalous using percentile attribution: "
                "shows which numeric features are most extreme relative to the dataset distribution. "
                "Use after anomaly_scan when the user asks why a row is flagged or wants to understand an outlier."
            ),
            explain_anomaly,
            ExplainAnomalyArgs,
        )
    )
    r.register(Tool("kmeans_clusters", "Cluster rows using KMeans on numeric columns.", kmeans_clusters, KMeansClustersArgs))
    r.register(Tool("simple_bar_spec", "Generate a bar chart spec from x/y columns.", simple_bar_spec, SimpleBarSpecArgs))
    r.register(Tool("histogram_spec", "Generate a histogram spec showing the distribution of a numeric column.", histogram_spec, HistogramSpecArgs))
    r.register(Tool("line_spec", "Generate a line chart spec of y over x, useful for trends/time series.", line_spec, LineSpecArgs))
    r.register(Tool("scatter_spec", "Generate a scatter plot spec of y vs x, including their correlation.", scatter_spec, ScatterSpecArgs))

    # New quality tools
    r.register(Tool("data_quality_report", "Detailed stats incl. missing %, skewness, and percentiles.", data_quality_report, DataQualityReportArgs))
    r.register(Tool("missingness_matrix", "Columns with highest missing ratios.", missingness_matrix, MissingnessMatrixArgs))
    r.register(Tool("overrepresented_categories", "Find dominant values in a categorical column.", overrepresented_categories, OverrepresentedCategoriesArgs))
    r.register(Tool("skewed_features", "List numeric features with high skewness.", skewed_features, SkewedFeaturesArgs))
    r.register(
        Tool(
            "evaluate_ml_predictions",
            (
                "Evaluate ML prediction outputs using classification, regression, "
                "forecasting, probability-score, and precomputed metric diagnostics."
            ),
            evaluate_ml_predictions,
            EvaluateMlPredictionsArgs,
        )
    )

    # Supervised learning
    r.register(
        Tool(
            "train_supervised_model",
            (
                "Train and evaluate a baseline supervised learning model (classification or "
                "regression) on the dataset against a target column, then persist it for reuse."
            ),
            train_supervised_model,
            TrainSupervisedModelArgs,
        )
    )
    r.register(
        Tool(
            "score_with_model",
            "Score the current dataset using a previously trained model identified by model_id.",
            score_with_model,
            ScoreWithModelArgs,
        )
    )
    r.register(
        Tool(
            "evaluate_trained_model",
            (
                "Return the persisted holdout evaluation metrics and metadata for a stored trained "
                "model identified by model_id. Use when the user asks to evaluate a trained model."
            ),
            evaluate_trained_model,
            EvaluateTrainedModelArgs,
        )
    )
    r.register(
        Tool(
            "explain_model",
            (
                "Compute SHAP / permutation feature importance for a stored model to show which features "
                "drive predictions globally. Use after train_supervised_model or with any existing model_id."
            ),
            explain_model,
            ExplainModelArgs,
        )
    )
    r.register(
        Tool(
            "shap_explain_prediction",
            (
                "Compute per-row signed SHAP contributions for a single prediction from a stored model. "
                "Shows which features pushed the prediction up or down for a specific row. "
                "Use when the user asks why the model predicted a specific value or wants a local explanation."
            ),
            shap_explain_prediction,
            ExplainPredictionArgs,
        )
    )
    r.register(
        Tool(
            "forecast_with_model",
            (
                "Generate a multi-step autoregressive forecast using a stored regression model that was "
                "trained with temporal lag features. Returns predicted values with 90% prediction intervals "
                "and a line chart. Requires a model trained on a datetime column."
            ),
            forecast_with_model,
            ForecastWithModelArgs,
        )
    )
    r.register(
        Tool(
            "compute_pdp",
            (
                "Compute partial dependence plots (PDPs) for a stored model: shows how each feature's "
                "marginal change affects the predicted value or probability, averaged over the dataset. "
                "Use after training or when the user asks to understand feature effects or relationships."
            ),
            compute_pdp,
            ComputePdpArgs,
        )
    )

    # Broader automated analysis
    r.register(
        Tool(
            "correlation_analysis",
            "Find and rank the strongest numeric correlations and categorical/numeric associations in the dataset.",
            correlation_analysis,
            CorrelationAnalysisArgs,
        )
    )
    r.register(
        Tool(
            "trend_analysis",
            "Analyze trend, period-over-period change, and peak/trough over a detected or specified datetime column.",
            trend_analysis,
            TrendAnalysisArgs,
        )
    )
    r.register(
        Tool(
            "auto_insights",
            (
                "Run data quality, relationship, anomaly, and trend analyses together and synthesize "
                "the most notable findings into one ranked summary."
            ),
            auto_insights,
            AutoInsightsArgs,
        )
    )

    r.register(
        Tool(
            "estimate_causal_effect",
            (
                "Estimate the causal effect of a treatment column on an outcome column using OLS regression "
                "with HC3 robust standard errors. Optionally controls for confounders and decomposes the "
                "effect via a mediator column. Returns ATE, 95% CI, p-value, effect size, and E-value "
                "(sensitivity to unmeasured confounders). Use when the user asks 'what causes X', "
                "'what is the effect of Y on Z', or 'does X drive Y'."
            ),
            estimate_causal_effect,
            CausalEffectArgs,
        )
    )
    r.register(
        Tool(
            "cross_dataset_profile",
            (
                "Discover join keys between the active dataset and other loaded datasets, then compute "
                "cross-dataset numeric correlations on the matched rows. Use when the user asks to "
                "compare datasets, find relationships across tables, or join and analyse multiple datasets."
            ),
            cross_dataset_profile,
            CrossDatasetProfileArgs,
        )
    )

    r.register(
        Tool(
            "hypothesis_test",
            (
                "Run a statistical hypothesis test. Supports: two_sample_t (Welch's t-test comparing two groups), "
                "one_sample_t (test if column mean equals a value), paired_t (before/after comparison), "
                "mannwhitney (non-parametric two-group comparison), chi_squared (independence between two categoricals), "
                "anova (compare means across 3+ groups), correlation (Pearson+Spearman significance test), "
                "power_analysis (compute required sample size or achieved power for a given effect size). "
                "Use when the user asks about statistical significance, t-test, chi-squared, ANOVA, "
                "sample size, statistical power, or whether a difference is significant."
            ),
            hypothesis_test,
            HypothesisTestArgs,
        )
    )

    _registry = r
    return _registry
