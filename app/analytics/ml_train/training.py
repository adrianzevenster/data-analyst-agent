from __future__ import annotations

import re
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.analytics.ml_eval.classification import evaluate_classification
from app.analytics.ml_eval.regression import evaluate_regression_or_forecast
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import build_preprocessor, split_feature_types

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBRegressor = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None  # type: ignore[assignment,misc]
    LGBMRegressor = None  # type: ignore[assignment,misc]

TaskHint = Literal["auto", "classification", "regression"]
ModelType = Literal[
    "auto",
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "decision_tree_classifier",
    "knn_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "linear_regression",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "ridge_regression",
    "lasso_regression",
    "decision_tree_regressor",
    "knn_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    # Generic family names that don't commit to a task - useful when the
    # caller (rule-based planner or LLM) knows the desired model family but
    # not yet whether the target makes this classification or regression.
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "knn",
    "xgboost",
    "lightgbm",
]

# Generic family alias -> {task_type: specific model_type}.
MODEL_FAMILY_ALIASES: dict[str, dict[str, str]] = {
    "random_forest": {"classification": "random_forest_classifier", "regression": "random_forest_regressor"},
    "gradient_boosting": {
        "classification": "gradient_boosting_classifier",
        "regression": "gradient_boosting_regressor",
    },
    "decision_tree": {"classification": "decision_tree_classifier", "regression": "decision_tree_regressor"},
    "knn": {"classification": "knn_classifier", "regression": "knn_regressor"},
    "xgboost": {"classification": "xgboost_classifier", "regression": "xgboost_regressor"},
    "lightgbm": {"classification": "lightgbm_classifier", "regression": "lightgbm_regressor"},
}

FEATURE_IMPORTANCE_TOP_N = 15

_PARAM_GRIDS: dict[str, dict] = {
    "random_forest_classifier": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 6, 12, 20], "model__min_samples_leaf": [1, 2, 4]},
    "random_forest_regressor": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 6, 12, 20], "model__min_samples_leaf": [1, 2, 4]},
    "gradient_boosting_classifier": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.05, 0.1, 0.2]},
    "gradient_boosting_regressor": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.05, 0.1, 0.2]},
    "xgboost_classifier": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.05, 0.1, 0.2]},
    "xgboost_regressor": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm_classifier": {"model__n_estimators": [100, 200], "model__max_depth": [-1, 5, 10], "model__learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm_regressor": {"model__n_estimators": [100, 200], "model__max_depth": [-1, 5, 10], "model__learning_rate": [0.05, 0.1, 0.2]},
}
HPARAM_N_ITER = 10
HPARAM_CV = 3

# Tree/ensemble classifiers that benefit from Platt calibration.
# Logistic regression is already calibrated; KNN probabilities are poor fits for Platt.
_CALIBRATION_CLASSIFIERS = frozenset({
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "decision_tree_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
})


def _require_installed(cls: Any, package_name: str) -> Any:
    if cls is None:
        raise ImportError(
            f"{package_name} is not installed in this environment. Add it to requirements-api.txt to use this model_type."
        )
    return cls


# Builders are lazy (lambdas) so an unavailable optional dependency (xgboost/
# lightgbm) only raises when that specific model_type is actually requested,
# not at import time for every other model_type.
CLASSIFIER_BUILDERS: dict[str, Callable[[], Any]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest_classifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "gradient_boosting_classifier": lambda: GradientBoostingClassifier(random_state=42),  # no class_weight support
    "decision_tree_classifier": lambda: DecisionTreeClassifier(random_state=42, max_depth=8, class_weight="balanced"),
    "knn_classifier": lambda: KNeighborsClassifier(),  # no class_weight support
    "xgboost_classifier": lambda: _require_installed(XGBClassifier, "xgboost")(
        eval_metric="logloss", random_state=42
    ),  # uses scale_pos_weight, not class_weight
    "lightgbm_classifier": lambda: _require_installed(LGBMClassifier, "lightgbm")(random_state=42, verbosity=-1, class_weight="balanced"),
}

REGRESSOR_BUILDERS: dict[str, Callable[[], Any]] = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting_regressor": lambda: GradientBoostingRegressor(random_state=42),
    "ridge_regression": lambda: Ridge(),
    "lasso_regression": lambda: Lasso(),
    "decision_tree_regressor": lambda: DecisionTreeRegressor(random_state=42, max_depth=8),
    "knn_regressor": lambda: KNeighborsRegressor(),
    "xgboost_regressor": lambda: _require_installed(XGBRegressor, "xgboost")(random_state=42),
    "lightgbm_regressor": lambda: _require_installed(LGBMRegressor, "lightgbm")(random_state=42, verbosity=-1),
}


_ROW_NUM_RE = re.compile(
    r'^(row[_\s]?(num(ber)?|id|index|no)|rownum|rowid|rowindex|'
    r'record[_\s]?(num(ber)?|id|no)|line[_\s]?(num(ber)?|no)|'
    r'seq(uence)?[_\s]?num(ber)?)$',
    re.IGNORECASE,
)


def _looks_like_id_col(df: pd.DataFrame, col: str) -> bool:
    """True when a numeric column is an identifier or sequential row number."""
    name = col.strip().lower().replace(" ", "_").replace("-", "_")
    if _ROW_NUM_RE.match(name):
        return True
    # Integer column where every value is unique and consecutive (1,2,3…N)
    if pd.api.types.is_integer_dtype(df[col]):
        s = df[col].dropna()
        n = len(s)
        if n > 1 and s.nunique() == n:
            vals = s.sort_values().to_numpy()
            if int(vals[-1]) - int(vals[0]) == n - 1:
                return True
    return False


def _should_log_transform(y: pd.Series) -> bool:
    """True when a regression target is non-negative and skewness > 1.5."""
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty or float(numeric.min()) < 0:
        return False
    try:
        return float(numeric.skew()) > 1.5
    except Exception:
        return False


def _infer_task_type(y: pd.Series, task_hint: TaskHint) -> str:
    if task_hint != "auto":
        return task_hint

    numeric = pd.to_numeric(y, errors="coerce").notna().mean() >= 0.90
    if numeric:
        unique = pd.to_numeric(y, errors="coerce").nunique(dropna=True)
        return "classification" if unique <= 20 else "regression"

    return "classification"


def _resolve_model_type(task_type: str, model_type: ModelType) -> str:
    if model_type == "auto":
        return "logistic_regression" if task_type == "classification" else "linear_regression"
    if model_type in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[model_type][task_type]
    return model_type


def _build_estimator(task_type: str, model_type: str):
    builders = CLASSIFIER_BUILDERS if task_type == "classification" else REGRESSOR_BUILDERS
    if model_type not in builders:
        valid = ", ".join(sorted(builders))
        raise ValueError(
            f"model_type {model_type!r} is not valid for task_type {task_type!r}. Valid options: {valid}"
        )
    return builders[model_type]()


def _extract_feature_importance(pipeline: Pipeline, top_n: int = FEATURE_IMPORTANCE_TOP_N) -> list[dict]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef)
    else:
        return []

    if len(importances) != len(feature_names):
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda item: -abs(item[1]))[:top_n]
    return [{"feature": str(name), "importance": float(value)} for name, value in pairs]


def _readout(task_type: str, model_type: str, n_train: int, n_test: int, model_id: str, evaluation: dict) -> str:
    if task_type == "classification":
        accuracy = evaluation.get("accuracy")
        metric = f"accuracy={accuracy:.4f}" if accuracy is not None else "metrics computed"
    else:
        wmape = evaluation.get("wmape")
        metric = f"WMAPE={wmape:.4f}" if wmape is not None else "metrics computed"

    return (
        f"Trained {model_type} ({task_type}) on {n_train} rows, evaluated on {n_test} held-out rows "
        f"({metric}). Model persisted as {model_id}."
    )


def train_supervised_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_hint: TaskHint = "auto",
    model_type: ModelType = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune: bool = True,
    dataset_id: str | None = None,
    model_manager: ModelManager | None = None,
) -> dict:
    feature_cols = [c for c in (feature_cols or df.columns) if c != target_col]

    d = df[[target_col] + feature_cols].dropna(subset=[target_col])
    if d.empty:
        return {"error": "No non-null rows for the target column."}

    # Drop identifier / sequential columns — they cause spurious correlations
    auto_dropped_id_cols = [c for c in feature_cols if _looks_like_id_col(d, c)]
    if auto_dropped_id_cols:
        feature_cols = [c for c in feature_cols if c not in set(auto_dropped_id_cols)]

    task_type = _infer_task_type(d[target_col], task_hint)
    resolved_model_type = _resolve_model_type(task_type, model_type)

    # Log-transform heavily skewed non-negative regression targets
    log_transform_target = False
    if task_type == "regression" and _should_log_transform(d[target_col]):
        log_transform_target = True
        d = d.copy()
        d[target_col] = np.log1p(d[target_col].astype(float))

    imbalance_ratio: float | None = None
    if task_type == "classification":
        counts = d[target_col].value_counts()
        if len(counts) >= 2 and counts.min() > 0:
            imbalance_ratio = round(float(counts.max() / counts.min()), 2)

    try:
        estimator = _build_estimator(task_type, resolved_model_type)
    except (ValueError, ImportError) as exc:
        return {"error": str(exc)}

    numeric_cols, categorical_cols, ordinal_cols, dropped_cols = split_feature_types(d, feature_cols)
    usable_features = numeric_cols + categorical_cols + ordinal_cols
    if not usable_features:
        return {"error": "No usable feature columns after excluding high-cardinality/unsupported columns."}

    X = d[usable_features]
    y = d[target_col]

    stratify = None
    if task_type == "classification" and y.nunique(dropna=True) >= 2 and y.value_counts().min() >= 2:
        stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        # Falls back when stratification isn't feasible (e.g. a class too small to split).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, ordinal_cols)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    best_params: dict | None = None
    param_grid = _PARAM_GRIDS.get(resolved_model_type) if tune else None
    if param_grid and len(X_train) >= 50:
        scoring = "f1_weighted" if task_type == "classification" else "neg_mean_absolute_percentage_error"
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=HPARAM_N_ITER, cv=HPARAM_CV,
            scoring=scoring, random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        pipeline.fit(X_train, y_train)

    # Apply Platt scaling calibration for tree/ensemble classifiers when HPO was
    # skipped (HPO's internal CV already regularises well enough to not need it).
    calibrated = False
    if (
        task_type == "classification"
        and resolved_model_type in _CALIBRATION_CLASSIFIERS
        and best_params is None
        and len(X_train) >= 100
    ):
        try:
            cal = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
            cal.fit(X_train, y_train)
            pipeline = cal
            calibrated = True
        except Exception:
            pass

    cv_result: dict | None = None
    if cv_folds >= 2:
        scoring = "f1_weighted" if task_type == "classification" else "neg_mean_absolute_percentage_error"
        cv_pipeline = Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols, ordinal_cols)),
            ("model", _build_estimator(task_type, resolved_model_type)),
        ])
        try:
            scores = cross_val_score(cv_pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            cv_result = {
                "folds": cv_folds,
                "scoring": scoring,
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "scores": [round(float(s), 4) for s in scores],
            }
        except Exception:
            pass

    y_pred = pipeline.predict(X_test)
    if log_transform_target:
        # Return to original scale for evaluation so WMAPE/R² are interpretable
        y_pred = np.expm1(y_pred)
        y_test_eval = np.expm1(y_test.to_numpy())
    else:
        y_test_eval = y_test.to_numpy()
    eval_df = pd.DataFrame({"actual": y_test_eval, "prediction": y_pred})

    probability_col = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        if hasattr(pipeline, "classes_"):
            classes = list(pipeline.classes_)
        elif hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps["model"], "classes_", []))
        else:
            classes = []
        if len(classes) == 2:
            eval_df["probability"] = pipeline.predict_proba(X_test)[:, -1]
            probability_col = "probability"

    if task_type == "classification":
        evaluation = evaluate_classification(
            eval_df, actual_col="actual", prediction_col="prediction", probability_col=probability_col
        )
    else:
        evaluation = evaluate_regression_or_forecast(eval_df, actual_col="actual", prediction_col="prediction")

    # For CalibratedClassifierCV the underlying fitted pipeline is in .estimator
    fi_source = pipeline.estimator if calibrated else pipeline
    feature_importance = _extract_feature_importance(fi_source) if hasattr(fi_source, "named_steps") else []

    manager = model_manager or ModelManager()
    previous = manager.find_previous(dataset_id, target_col)
    meta = manager.save_model(
        pipeline,
        task_type=task_type,
        model_type=resolved_model_type,
        target_col=target_col,
        feature_cols=usable_features,
        dataset_id=dataset_id,
        log_transform_target=log_transform_target,
        evaluation=evaluation,
    )

    preprocessing_notes: list[str] = []
    if auto_dropped_id_cols:
        preprocessing_notes.append(
            f"Auto-excluded identifier column(s) from features: {', '.join(auto_dropped_id_cols)}."
        )
    if log_transform_target:
        preprocessing_notes.append(
            f"Target '{target_col}' was log-transformed (log1p) due to high skewness — "
            "metrics are reported in the original scale."
        )

    model_comparison: dict | None = None
    comparison_note = ""
    if previous and previous.evaluation:
        prev_eval = previous.evaluation
        if task_type == "classification":
            prev_v = prev_eval.get("accuracy")
            curr_v = evaluation.get("accuracy")
            metric = "accuracy"
            improved = bool((curr_v or 0.0) > (prev_v or 0.0))
        else:
            prev_v = prev_eval.get("wmape")
            curr_v = evaluation.get("wmape")
            metric = "wmape"
            improved = bool((curr_v or float("inf")) < (prev_v or float("inf")))

        if prev_v is not None and curr_v is not None:
            delta = round(float(curr_v) - float(prev_v), 4)
            model_comparison = {
                "previous_model_id": previous.model_id,
                "previous_model_type": previous.model_type,
                "metric": metric,
                "previous": round(float(prev_v), 4),
                "current": round(float(curr_v), 4),
                "delta": delta,
                "improved": improved,
            }
            arrow = "↑" if improved else "↓"
            comparison_note = (
                f" vs previous {previous.model_type}: "
                f"{metric.upper()} {round(float(prev_v), 4)} → {round(float(curr_v), 4)} "
                f"({arrow}{abs(delta):.4f})"
            )

    return {
        "model_id": meta.model_id,
        "task_type": task_type,
        "model_type": resolved_model_type,
        "target_col": target_col,
        "feature_cols": usable_features,
        "dropped_feature_cols": dropped_cols,
        "auto_dropped_id_cols": auto_dropped_id_cols,
        "log_transform_target": log_transform_target,
        "preprocessing_notes": preprocessing_notes,
        "n_rows_total": int(len(d)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "evaluation": evaluation,
        "cv": cv_result,
        "best_params": best_params,
        "imbalance_ratio": imbalance_ratio,
        "calibrated": calibrated,
        "feature_importance": feature_importance,
        "model_comparison": model_comparison,
        "engineering_readout": _readout(
            task_type, resolved_model_type, len(X_train), len(X_test), meta.model_id, evaluation
        ) + comparison_note,
    }
