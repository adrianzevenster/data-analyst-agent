from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.evaluation import evaluate_trained_model
from app.analytics.ml_train.scoring import score_with_model
from app.analytics.ml_train.training import train_supervised_model


def _classification_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
            "segment": rng.choice(["a", "b", "c"], n),
        }
    )
    df["churn"] = ((df["feature1"] + (df["segment"] == "a").astype(int)) > 0.3).astype(int)
    return df


def _regression_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"feature1": rng.normal(0, 1, n), "feature2": rng.normal(5, 2, n)})
    df["revenue"] = 10 * df["feature1"] + 2 * df["feature2"] + rng.normal(0, 0.1, n)
    return df


@pytest.fixture
def model_manager(tmp_path):
    return ModelManager(base_dir=str(tmp_path))


def test_train_classification_model_persists_and_evaluates(model_manager):
    df = _classification_df()

    result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    assert result["task_type"] == "classification"
    # auto model selection picks the best of 3 candidates — don't assert a specific type
    assert result["model_type"] in (
        "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier",
        "xgboost_classifier", "lightgbm_classifier",
    )
    assert result["n_rows_train"] + result["n_rows_test"] == result["n_rows_total"]
    assert result["evaluation"]["accuracy"] > 0.7
    assert result["feature_importance"]
    assert "Model persisted as" in result["engineering_readout"]

    # Model is actually retrievable from the manager it was saved through.
    pipeline, meta = model_manager.load_model(result["model_id"])
    assert meta.task_type == "classification"
    assert set(meta.feature_cols) == {"feature1", "feature2", "segment"}
    assert pipeline.predict(df[meta.feature_cols].head(1)) is not None


def test_train_regression_model_persists_and_evaluates(model_manager):
    df = _regression_df()

    result = train_supervised_model(
        df, target_col="revenue", task_hint="regression", model_manager=model_manager
    )

    assert result["task_type"] == "regression"
    # auto model selection picks the best of 3 candidates — don't assert a specific type
    assert result["model_type"] in (
        "linear_regression", "ridge_regression", "random_forest_regressor",
        "gradient_boosting_regressor", "xgboost_regressor", "lightgbm_regressor",
    )
    assert result["evaluation"]["r2"] > 0.9


def test_train_random_forest_classifier_uses_feature_importances(model_manager):
    df = _classification_df()

    result = train_supervised_model(
        df,
        target_col="churn",
        model_type="random_forest_classifier",
        model_manager=model_manager,
    )

    assert result["model_type"] == "random_forest_classifier"
    assert result["feature_importance"]
    # feature_importances_ values are non-negative, unlike linear coefficients.
    assert all(item["importance"] >= 0 for item in result["feature_importance"])


@pytest.mark.parametrize(
    "model_type",
    [
        "gradient_boosting_classifier",
        "decision_tree_classifier",
        "knn_classifier",
        "xgboost_classifier",
        "lightgbm_classifier",
    ],
)
def test_train_classification_supports_expanded_model_types(model_manager, model_type):
    df = _classification_df()

    result = train_supervised_model(df, target_col="churn", model_type=model_type, model_manager=model_manager)

    assert result["model_type"] == model_type
    assert "error" not in result
    assert result["evaluation"]["accuracy"] > 0.5

    pipeline, meta = model_manager.load_model(result["model_id"])
    assert pipeline.predict(df[meta.feature_cols].head(1)) is not None


@pytest.mark.parametrize(
    "model_type",
    [
        "ridge_regression",
        "lasso_regression",
        "gradient_boosting_regressor",
        "decision_tree_regressor",
        "knn_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    ],
)
def test_train_regression_supports_expanded_model_types(model_manager, model_type):
    df = _regression_df()

    result = train_supervised_model(
        df, target_col="revenue", task_hint="regression", model_type=model_type, model_manager=model_manager
    )

    assert result["model_type"] == model_type
    assert "error" not in result
    assert result["evaluation"]["r2"] > 0.5


def test_train_resolves_generic_family_alias_to_classifier_variant(model_manager):
    df = _classification_df()

    result = train_supervised_model(
        df, target_col="churn", model_type="random_forest", model_manager=model_manager
    )

    assert result["model_type"] == "random_forest_classifier"


def test_train_resolves_generic_family_alias_to_regressor_variant(model_manager):
    df = _regression_df()

    result = train_supervised_model(
        df,
        target_col="revenue",
        task_hint="regression",
        model_type="random_forest",
        model_manager=model_manager,
    )

    assert result["model_type"] == "random_forest_regressor"


def test_train_returns_error_for_model_type_mismatched_with_task(model_manager):
    df = _classification_df()

    result = train_supervised_model(
        df, target_col="churn", model_type="ridge_regression", model_manager=model_manager
    )

    assert "error" in result
    assert "ridge_regression" in result["error"]
    assert "classification" in result["error"]


def test_train_drops_high_cardinality_categorical_features(model_manager):
    df = _classification_df()
    df["customer_note"] = [f"note-{i}" for i in range(len(df))]  # unique per row

    result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    assert "customer_note" in result["dropped_feature_cols"]
    assert "customer_note" not in result["feature_cols"]


def test_train_returns_error_for_all_null_target(model_manager):
    df = _classification_df()
    df["churn"] = None

    result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    assert "error" in result


def test_train_returns_error_when_no_usable_features(model_manager):
    df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(100)],  # all-unique, exceeds cardinality cap -> dropped
            "target": [0, 1] * 50,
        }
    )

    result = train_supervised_model(df, target_col="target", feature_cols=["id"], model_manager=model_manager)

    assert "error" in result


def test_score_with_model_round_trip(model_manager):
    df = _classification_df()
    train_result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    score_result = score_with_model(df, model_id=train_result["model_id"], model_manager=model_manager)

    assert score_result["n_rows_scored"] == len(df)
    assert score_result["scored_rows"]
    assert "prediction" in score_result["scored_rows"][0]
    assert "prediction_probability" in score_result["scored_rows"][0]


def test_evaluate_trained_model_returns_persisted_metrics(model_manager):
    df = _classification_df()
    train_result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    eval_result = evaluate_trained_model(df, model_id=train_result["model_id"], model_manager=model_manager)

    assert eval_result["model_id"] == train_result["model_id"]
    assert eval_result["task_type"] == "classification"
    assert eval_result["target_col"] == "churn"
    assert eval_result["evaluation"] == train_result["evaluation"]
    assert "engineering_readout" in eval_result


def test_score_with_model_raises_on_missing_required_columns(model_manager):
    df = _classification_df()
    train_result = train_supervised_model(df, target_col="churn", model_manager=model_manager)

    incompatible_df = df.drop(columns=["segment"])

    with pytest.raises(ValueError, match="missing columns"):
        score_with_model(incompatible_df, model_id=train_result["model_id"], model_manager=model_manager)


def test_model_manager_unknown_model_id_raises_key_error(model_manager):
    with pytest.raises(KeyError):
        model_manager.load_model("does-not-exist")


def test_train_with_text_column_encodes_and_trains(model_manager):
    """High-cardinality text columns should be embedded rather than dropped.

    The review column has n unique strings (unique fraction > 0.5, so OrdinalEncoder
    also rejects it) with mean word count > 3, triggering TextEmbeddingEncoder.
    """
    rng = np.random.default_rng(42)
    n = 200
    adjectives = ["excellent", "poor", "average", "terrible", "outstanding",
                  "mediocre", "superb", "awful", "decent", "fantastic"]
    nouns = ["quality", "service", "delivery", "packaging", "experience",
             "product", "support", "value", "performance", "design"]
    # n unique sentences — unique fraction = 1.0, far above the 0.5 ordinal cap
    reviews = [
        f"review {i} the {rng.choice(adjectives)} {rng.choice(nouns)} was really {rng.choice(adjectives)} overall"
        for i in range(n)
    ]
    df = pd.DataFrame({
        "review": reviews,
        "score": rng.integers(1, 6, n),
    })
    df["positive"] = (df["score"] >= 4).astype(int)

    result = train_supervised_model(
        df, target_col="positive", feature_cols=["review", "score"],
        model_manager=model_manager,
    )

    assert "error" not in result, result.get("error")
    assert "review" in result["text_feature_cols"]
    assert result["evaluation"]["accuracy"] > 0.5
    # Round-trip: the serialised pipeline must handle text at inference time.
    pipeline, meta = model_manager.load_model(result["model_id"])
    preds = pipeline.predict(df[meta.feature_cols].head(5))
    assert len(preds) == 5


# ---------------------------------------------------------------------------
# Lag / rolling feature engineering
# ---------------------------------------------------------------------------

def test_lag_features_created_for_temporal_regression(model_manager):
    """Datasets with a datetime sort column and a regression target should have
    lag/rolling features auto-created and surfaced in lag_feature_cols."""
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    values = np.cumsum(rng.normal(0, 1, n)) + 50
    df = pd.DataFrame({
        "date": dates,
        "value": values,
        "noise": rng.normal(0, 0.1, n),
    })
    df["target"] = df["value"].shift(-1).fillna(method="ffill")

    result = train_supervised_model(
        df, target_col="target", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )

    assert "error" not in result, result.get("error")
    assert result["lag_feature_cols"], "Expected lag feature columns to be created"
    # Lag cols should reference the numeric source columns
    assert any("__lag_" in c for c in result["lag_feature_cols"])
    assert any("__roll_mean_" in c for c in result["lag_feature_cols"])
    # lag_config must be persisted so scoring can re-apply them
    _, meta = model_manager.load_model(result["model_id"])
    assert meta.lag_config is not None
    assert meta.lag_config["sort_col"] == "date"


def test_lag_features_round_trip_scoring(model_manager):
    """score_with_model must re-apply lag features and score all rows."""
    rng = np.random.default_rng(8)
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    values = np.cumsum(rng.normal(0, 1, n)) + 20
    df = pd.DataFrame({
        "date": dates,
        "value": values,
    })
    df["target"] = df["value"].shift(-1).fillna(method="ffill")

    train_result = train_supervised_model(
        df, target_col="target", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )
    assert "error" not in train_result

    score_result = score_with_model(df, model_id=train_result["model_id"], model_manager=model_manager)
    assert score_result["n_rows_scored"] > 0
    assert "prediction" in score_result["scored_rows"][0]


# ---------------------------------------------------------------------------
# Target encoding for high-cardinality categoricals
# ---------------------------------------------------------------------------

def test_target_encoder_used_for_ordinal_categorical(model_manager):
    """Columns with 51-500 unique values (ordinal bucket) should be processed
    via TargetEncoder, producing meaningful regression predictions."""
    rng = np.random.default_rng(9)
    n = 300
    # 80 unique zip codes → exceeds OHE cap (50), below ordinal cap (500)
    zip_codes = [f"ZIP{i:04d}" for i in rng.integers(0, 80, n)]
    df = pd.DataFrame({
        "zip_code": zip_codes,
        "area_sqft": rng.normal(1500, 300, n),
        "price": rng.normal(300_000, 50_000, n),
    })
    # Make zip_code predictive: zip groups with index < 40 → higher price
    zip_idx = np.array([int(z[3:]) for z in zip_codes])
    df["price"] += np.where(zip_idx < 40, 50_000, -50_000)

    result = train_supervised_model(
        df, target_col="price", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )

    assert "error" not in result, result.get("error")
    # zip_code should appear in feature_cols (not dropped)
    assert "zip_code" in result["feature_cols"]
    # TargetEncoder encodes the predictive signal — R² should be positive
    assert result["evaluation"]["r2"] > 0.2


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def test_interaction_features_added_for_wide_numeric_datasets(model_manager):
    """Datasets with ≥3 numeric features and ≥200 rows should trigger interaction
    feature creation, surfaced via interaction_features_added=True."""
    rng = np.random.default_rng(11)
    n = 250
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.normal(-3, 1.5, n),
        "d": rng.normal(10, 3, n),
    })
    # Make target depend on an interaction term (a*b) to verify it helps
    df["target"] = 2 * df["a"] * df["b"] + 0.5 * df["c"] + rng.normal(0, 0.1, n)

    result = train_supervised_model(
        df, target_col="target", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )

    assert "error" not in result, result.get("error")
    assert result["interaction_features_added"] is True
    # Model should exploit the a*b interaction — expect R² > 0.8
    assert result["evaluation"]["r2"] > 0.8


def test_interaction_features_not_added_for_small_datasets(model_manager):
    """Datasets with <200 rows should NOT trigger interaction features."""
    df = _regression_df(n=100)

    result = train_supervised_model(
        df, target_col="revenue", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )

    assert "error" not in result
    assert result["interaction_features_added"] is False


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def test_baseline_comparison_returned_for_classification(model_manager):
    """train_supervised_model should always return baseline_comparison for classification."""
    df = _classification_df(n=200)
    result = train_supervised_model(
        df, target_col="churn", task_hint="classification",
        model_type="logistic_regression", model_manager=model_manager,
    )
    assert "error" not in result, result.get("error")
    bc = result.get("baseline_comparison")
    assert bc is not None
    assert "baselines" in bc
    assert bc["primary_metric"] == "accuracy"
    assert bc["best_baseline_metric"] is not None
    assert bc["beats_baseline"] is not None
    assert bc["delta"] is not None
    # Logistic regression should beat a majority-class baseline on this dataset
    assert bc["beats_baseline"] is True


def test_baseline_comparison_returned_for_regression(model_manager):
    """train_supervised_model should return WMAPE-based baseline for regression."""
    df = _regression_df(n=200)
    result = train_supervised_model(
        df, target_col="revenue", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )
    assert "error" not in result
    bc = result.get("baseline_comparison")
    assert bc is not None
    assert bc["primary_metric"] == "wmape"
    assert "mean" in bc["baselines"]
    assert "median" in bc["baselines"]


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def test_leakage_detection_flags_correlated_feature(model_manager):
    """A feature that is almost identical to the target should be flagged as high risk."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "target": target,
        "leaky_col": target + rng.normal(0, 0.01, n),  # r ≈ 0.9999
        "safe_col": rng.normal(0, 1, n),
    })
    result = train_supervised_model(
        df, target_col="target", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )
    assert "error" not in result
    warnings = result.get("leakage_warnings", [])
    high_risk = [w for w in warnings if w["risk"] == "high"]
    assert any(w["feature"] == "leaky_col" for w in high_risk), (
        f"Expected leaky_col in high-risk warnings; got {warnings}"
    )
    assert all(w["feature"] != "safe_col" for w in high_risk)


def test_train_respects_max_rows(model_manager):
    df = _regression_df(n=250, seed=123)

    result = train_supervised_model(
        df,
        target_col="revenue",
        task_hint="regression",
        model_type="ridge_regression",
        tune=False,
        cv_folds=0,
        max_rows=120,
        model_manager=model_manager,
    )

    assert "error" not in result
    assert result["sampled_rows"] is True
    assert result["n_rows_source"] == 250
    assert result["n_rows_total"] == 120
    assert result["max_rows"] == 120


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def test_drift_detection_reports_none_when_distributions_match(model_manager):
    """Scoring data from the same distribution should show drift_severity='none'."""
    df = _regression_df(n=300, seed=7)
    result = train_supervised_model(
        df, target_col="revenue", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )
    assert "error" not in result

    from app.analytics.ml_train.scoring import score_with_model
    score_result = score_with_model(df, model_id=result["model_id"], model_manager=model_manager)
    drift = score_result.get("drift")
    assert drift is not None
    assert drift["overall_severity"] == "none", (
        f"Expected no drift when using training distribution; got {drift}"
    )


def test_drift_detection_flags_shifted_distribution(model_manager):
    """Scoring data with a major mean shift should be flagged by drift detection."""
    df_train = _regression_df(n=300, seed=99)
    result = train_supervised_model(
        df_train, target_col="revenue", task_hint="regression",
        model_type="ridge_regression", model_manager=model_manager,
    )
    assert "error" not in result

    # Simulate severe distribution shift: multiply numeric features by 100
    df_score = df_train.copy()
    for col in df_score.select_dtypes("number").columns:
        if col != "revenue":
            df_score[col] = df_score[col] * 100

    from app.analytics.ml_train.scoring import score_with_model
    score_result = score_with_model(df_score, model_id=result["model_id"], model_manager=model_manager)
    drift = score_result.get("drift")
    assert drift is not None
    assert drift["overall_severity"] != "none", (
        f"Expected drift to be detected after 100× mean shift; got {drift}"
    )
    assert drift["n_drifted"] > 0


# ---------------------------------------------------------------------------
# LightGBM in auto-select candidate pool
# ---------------------------------------------------------------------------

def test_lightgbm_in_auto_candidates_when_installed():
    from app.analytics.ml_train.training import _AUTO_CANDIDATES, LGBMClassifier
    if LGBMClassifier is None:
        pytest.skip("LightGBM not installed")
    assert "lightgbm_classifier" in _AUTO_CANDIDATES["classification"]
    assert "lightgbm_regressor" in _AUTO_CANDIDATES["regression"]


def test_auto_candidates_always_has_at_least_two_per_task():
    from app.analytics.ml_train.training import _AUTO_CANDIDATES
    assert len(_AUTO_CANDIDATES["classification"]) >= 2
    assert len(_AUTO_CANDIDATES["regression"]) >= 2


def test_auto_select_chooses_among_available_candidates(model_manager):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    result = train_supervised_model(
        df, target_col="label", model_type="auto",
        tune=False, cv_folds=3, model_manager=model_manager,
    )
    assert "error" not in result
    note = " ".join(result.get("preprocessing_notes", []))
    assert "auto-selected" in note.lower() or "auto" in result.get("model_type", "")
