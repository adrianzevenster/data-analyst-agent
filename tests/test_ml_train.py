from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
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
