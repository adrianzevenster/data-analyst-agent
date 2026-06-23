"""Golden-query regression suite for the rule-based planner's tool selection.

Each case pins a (message, dataset) pair to the tool names the planner must
choose. This is the signal that catches "the planner stopped picking X for a
prompt that always used to trigger X" when the rule list or keyword sets
change.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager
from app.analytics.ml_train.model_store import ModelManager

GENERIC_DF = pd.DataFrame(
    {
        "region": ["east", "west", "east", "west"],
        "revenue": [100.0, 200.0, 150.0, 50.0],
        "units": [1, 2, 3, 4],
    }
)

TAXI_DF = pd.DataFrame(
    {
        "trip_duration_sec": [420, 610, 380, 900],
        "distance_traveled_Km": [2.1, 4.8, 1.6, 7.3],
        "wait_time_cost": [0.5, 1.0, 0.0, 2.0],
        "distance_cost": [4.2, 9.6, 3.2, 14.6],
        "fare_w_flag": [5.7, 11.1, 4.7, 17.1],
        "tip": [1.0, 2.0, 0.5, 3.0],
        "miscellaneous_fees": [0.0, 0.5, 0.0, 1.0],
        "total_fare_new": [6.7, 13.6, 5.2, 21.1],
        "num_of_passengers": [1, 2, 1, 3],
        "surge_applied": [0, 1, 0, 1],
    }
)

CLASSIFICATION_DF = pd.DataFrame(
    {
        "actual": [1, 0, 1, 0],
        "prediction": [1, 1, 1, 0],
        "probability": [0.9, 0.7, 0.6, 0.2],
    }
)

REGRESSION_DF = pd.DataFrame(
    {
        "actual": [100.0, 200.0, 150.5, 300.25, 80.0, 412.0, 90.5, 175.0, 60.0, 220.0,
                   130.0, 145.0, 310.0, 95.0, 205.0, 88.0, 410.0, 99.0, 178.0, 65.0,
                   225.0],
        "prediction": [110.0, 190.0, 140.0, 295.0, 85.0, 400.0, 92.0, 170.0, 58.0, 215.0,
                       128.0, 150.0, 305.0, 90.0, 210.0, 86.0, 405.0, 101.0, 180.0, 63.0,
                       222.0],
    }
)

SCORED_PREDICTIONS_DF = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "churn_prediction": [1, 0, 1],
        "probability": [0.9, 0.2, 0.6],
    }
)


GOLDEN_CASES = [
    pytest.param("Give me a profile of this dataset", GENERIC_DF, ["profile_dataset"], id="profile"),
    pytest.param("Show me the schema and overview", GENERIC_DF, ["profile_dataset"], id="overview"),
    pytest.param("Run a data quality healthcheck", GENERIC_DF, ["data_quality_report"], id="quality"),
    # "columns" also matches the profile_dataset keyword set, so both fire.
    pytest.param(
        "Which columns have missing values?",
        GENERIC_DF,
        ["profile_dataset", "missingness_matrix"],
        id="missingness",
    ),
    pytest.param("Are any features skewed?", GENERIC_DF, ["skewed_features"], id="skew"),
    pytest.param(
        "Check for class imbalance in column region",
        GENERIC_DF,
        ["overrepresented_categories"],
        id="overrepresented-with-column",
    ),
    # "for" is too common in natural phrasing to treat as a column marker, and
    # "class" isn't an actual column here, so this must fall back rather than
    # misfire on overrepresented_categories with a bogus column.
    pytest.param(
        "Check for class imbalance",
        GENERIC_DF,
        ["data_quality_report"],
        id="overrepresented-no-column",
    ),
    pytest.param("Give me a breakdown by region", GENERIC_DF, ["multidim_pivot"], id="pivot"),
    pytest.param("sql: select * from t", GENERIC_DF, ["duckdb_query"], id="sql"),
    pytest.param("Find outliers in this data", GENERIC_DF, ["anomaly_scan"], id="anomaly"),
    pytest.param("Cluster these rows into segments", GENERIC_DF, ["kmeans_clusters"], id="cluster"),
    pytest.param(
        "Evaluate model performance",
        CLASSIFICATION_DF,
        ["evaluate_ml_predictions"],
        id="ml-eval-classification-keyword",
    ),
    pytest.param(
        "What's going on with this data?",
        CLASSIFICATION_DF,
        ["evaluate_ml_predictions"],
        id="ml-eval-classification-auto-detect",
    ),
    pytest.param(
        "What's going on with this data?",
        REGRESSION_DF,
        ["evaluate_ml_predictions"],
        id="ml-eval-regression-auto-detect",
    ),
    pytest.param(
        "What's going on with this data?",
        SCORED_PREDICTIONS_DF,
        ["evaluate_ml_predictions"],
        id="ml-eval-scored-predictions-auto-detect",
    ),
    # "Tell me about this" triggers the full analyse sweep (profile + quality +
    # insights + correlations) rather than just auto_insights, so an ambiguous
    # question gets a comprehensive answer.
    pytest.param(
        "Tell me about this file",
        GENERIC_DF,
        ["profile_dataset", "data_quality_report", "auto_insights", "correlation_analysis", "kmeans_clusters", "anomaly_scan"],
        id="fallback-no-keywords-with-dataset",
    ),
    pytest.param(
        "Train a model to predict revenue",
        GENERIC_DF,
        ["train_supervised_model"],
        id="train-with-target",
    ),
    # No nameable target column: must not guess one. Profiles the dataset
    # instead so the user can see available columns and name a target.
    pytest.param("Train a model", GENERIC_DF, ["profile_dataset"], id="train-without-target"),
    pytest.param(
        "score with model 3aabadbc-b396-451b-b28f-b166482cce79",
        GENERIC_DF,
        ["score_with_model"],
        id="score-with-model-id",
    ),
    # No model_id named: must not guess, falls through to generic fallback.
    pytest.param("score with model please", GENERIC_DF, ["auto_insights"], id="score-without-model-id"),
    pytest.param("What correlations exist in this data?", GENERIC_DF, ["correlation_analysis"], id="correlation"),
    pytest.param("Show me the trend over time", GENERIC_DF, ["trend_analysis"], id="trend"),
    pytest.param("What insights stand out in this data?", GENERIC_DF, ["auto_insights"], id="explicit-insights"),
]


@pytest.fixture
def planner(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_enabled", False)
    manager = DatasetManager(base_dir=str(tmp_path))
    p = Planner()
    p.dm = manager
    p.model_manager = ModelManager(base_dir=str(tmp_path))
    return p, manager


@pytest.mark.parametrize("message, df, expected_tools", GOLDEN_CASES)
def test_planner_selects_expected_tools(planner, message, df, expected_tools):
    p, manager = planner
    dataset_id = None
    if df is not None:
        dataset_id = manager.register_df(df, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan(message, dataset_id)

    assert [c.name for c in calls] == expected_tools


def test_ml_eval_task_hint_matches_detected_task_type(planner):
    p, manager = planner
    dataset_id = manager.register_df(REGRESSION_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan("evaluate model performance", dataset_id)

    assert calls[0].name == "evaluate_ml_predictions"
    assert calls[0].arguments["task_hint"] == "regression"


def test_train_taxi_fare_request_infers_total_fare_target(planner):
    p, manager = planner
    dataset_id = manager.register_df(TAXI_DF, "taxi.csv").dataset_id

    calls, _, _, _, _ = p.plan("Build a model for me to predict taxi fare", dataset_id)

    assert [c.name for c in calls] == ["train_supervised_model"]
    assert calls[0].arguments["target_col"] == "total_fare_new"


def test_overrepresented_extracts_the_correct_named_column(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan("Check for class imbalance in column region", dataset_id)

    assert calls[0].name == "overrepresented_categories"
    assert calls[0].arguments["col"] == "region"


def test_overrepresented_matches_bare_in_phrasing_against_real_column(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan("Check for dominant values in region", dataset_id)

    assert calls[0].name == "overrepresented_categories"
    assert calls[0].arguments["col"] == "region"


def test_score_with_model_falls_back_to_most_recently_trained_model(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan(
        "score with model please",
        dataset_id,
        trained_model_ids=["older-id", "newest-id"],
    )

    assert calls[0].name == "score_with_model"
    assert calls[0].arguments["model_id"] == "newest-id"


def test_score_with_model_without_history_or_explicit_id_falls_through(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan("score with model please", dataset_id, trained_model_ids=[])

    assert [c.name for c in calls] == ["auto_insights"]


def test_evaluate_this_model_uses_most_recently_trained_model(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan(
        "Evaluate this model",
        dataset_id,
        trained_model_ids=["older-id", "newest-id"],
    )

    assert calls[0].name == "evaluate_trained_model"
    assert calls[0].arguments["model_id"] == "newest-id"


def test_evaluate_model_uses_explicit_model_id(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id
    model_id = "123e4567-e89b-12d3-a456-426614174000"

    calls, _, _, _, _ = p.plan(f"Evaluate model {model_id}", dataset_id)

    assert calls[0].name == "evaluate_trained_model"
    assert calls[0].arguments["model_id"] == model_id


def test_evaluate_this_model_uses_latest_stored_model_for_dataset(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id
    p.model_manager.save_model(
        {"kind": "dummy"},
        model_id="older-model",
        task_type="regression",
        model_type="linear_regression",
        target_col="revenue",
        feature_cols=["units"],
        dataset_id=dataset_id,
        evaluation={"wmape": 0.5},
    )
    p.model_manager.save_model(
        {"kind": "dummy"},
        model_id="newer-model",
        task_type="regression",
        model_type="ridge_regression",
        target_col="revenue",
        feature_cols=["units"],
        dataset_id=dataset_id,
        evaluation={"wmape": 0.4},
    )

    calls, _, _, _, _ = p.plan("Evaluate this model", dataset_id, trained_model_ids=[])

    assert calls[0].name == "evaluate_trained_model"
    assert calls[0].arguments["model_id"] == "newer-model"


@pytest.mark.parametrize(
    "message, expected_model_type",
    [
        ("Train a random forest to predict revenue", "random_forest"),
        ("Build an xgboost model to predict revenue", "xgboost"),
        ("Train a lightgbm model to predict revenue", "lightgbm"),
        ("Fit a gradient boosting model to predict revenue", "gradient_boosting"),
        ("Train a decision tree to predict revenue", "decision_tree"),
        ("Train a knn model to predict revenue", "knn"),
        ("Train a ridge regression to predict revenue", "ridge_regression"),
    ],
)
def test_train_picks_up_named_model_type_from_message(planner, message, expected_model_type):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan(message, dataset_id)

    assert calls[0].name == "train_supervised_model"
    assert calls[0].arguments["model_type"] == expected_model_type


def test_train_without_named_model_type_omits_model_type_arg(planner):
    p, manager = planner
    dataset_id = manager.register_df(GENERIC_DF, "dataset.csv").dataset_id

    calls, _, _, _, _ = p.plan("Train a model to predict revenue", dataset_id)

    assert calls[0].name == "train_supervised_model"
    assert "model_type" not in calls[0].arguments
