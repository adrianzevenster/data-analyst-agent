from __future__ import annotations

import os
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
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.analytics.ml_eval.classification import evaluate_classification
from app.analytics.ml_eval.regression import evaluate_regression_or_forecast
from app.analytics.ml_train.baseline import compute_baselines, finalise_baseline_comparison
from app.analytics.ml_train.drift import compute_training_stats
from app.analytics.ml_train.experiment_tracker import get_tracker
from app.analytics.ml_train.leakage import detect_leakage
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.onnx_export import try_export_onnx
from app.analytics.ml_train.preprocessing import (
    LAG_DEFAULTS,
    ROLLING_DEFAULTS,
    build_preprocessor,
    engineer_lag_features,
    split_feature_types,
    TEXT_EMBEDDING_N_COMPONENTS,
)

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

# Cap parallelism so background training doesn't starve other processes on
# machines with many cores.
_N_JOBS = min(4, os.cpu_count() or 1)

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

# Candidates for auto model selection per task type.
# XGBoost is preferred over gradient_boosting when available (faster, often better).
_AUTO_CANDIDATES: dict[str, list[str]] = {
    "classification": (
        ["logistic_regression", "random_forest_classifier", "xgboost_classifier"]
        if XGBClassifier is not None
        else ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
    ),
    "regression": (
        ["ridge_regression", "random_forest_regressor", "xgboost_regressor"]
        if XGBRegressor is not None
        else ["ridge_regression", "random_forest_regressor", "gradient_boosting_regressor"]
    ),
}
# Max rows to evaluate each candidate on — ranking needs relative ordering,
# not precise absolute scores, so a sample is enough.
_AUTO_SELECT_SAMPLE = 2000


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
    "random_forest_classifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=_N_JOBS, class_weight="balanced"),
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
    "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=_N_JOBS),
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
        # Sentinel — caller must run _auto_select_model first.
        return "auto"
    if model_type in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[model_type][task_type]
    return model_type


def _build_estimator(task_type: str, model_type: str, scale_pos_weight: float | None = None):
    builders = CLASSIFIER_BUILDERS if task_type == "classification" else REGRESSOR_BUILDERS
    if model_type not in builders:
        valid = ", ".join(sorted(builders))
        raise ValueError(
            f"model_type {model_type!r} is not valid for task_type {task_type!r}. Valid options: {valid}"
        )
    estimator = builders[model_type]()
    if (
        scale_pos_weight is not None
        and model_type == "xgboost_classifier"
        and XGBClassifier is not None
        and isinstance(estimator, XGBClassifier)
    ):
        estimator.set_params(scale_pos_weight=scale_pos_weight)
    return estimator


def _auto_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: list[str],
    datetime_cols: list[str],
    text_cols: list[str] | None = None,
    scale_pos_weight: float | None = None,
) -> tuple[str, str]:
    """3-candidate CV shootout on a capped sample. Returns (best_model_type, note)."""
    candidates = _AUTO_CANDIDATES[task_type]
    scoring = "f1_weighted" if task_type == "classification" else "neg_mean_absolute_percentage_error"

    if len(X) > _AUTO_SELECT_SAMPLE:
        sample_idx = X.sample(n=_AUTO_SELECT_SAMPLE, random_state=42).index
        X_s, y_s = X.loc[sample_idx], y.loc[sample_idx]
    else:
        X_s, y_s = X, y

    best_type = candidates[0]
    best_score = float("-inf")
    scores: dict[str, float] = {}

    for candidate in candidates:
        try:
            pipe = Pipeline([
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols)),
                ("model", _build_estimator(task_type, candidate, scale_pos_weight=scale_pos_weight)),
            ])
            cv_scores = cross_val_score(pipe, X_s, y_s, cv=3, scoring=scoring, n_jobs=_N_JOBS)
            mean_score = float(cv_scores.mean())
            scores[candidate] = round(mean_score, 4)
            if mean_score > best_score:
                best_score = mean_score
                best_type = candidate
        except Exception:
            continue

    score_summary = "; ".join(f"{k}={v:.4f}" for k, v in scores.items())
    note = f"Auto-selected {best_type} via 3-fold CV ({scoring}: {score_summary})"
    return best_type, note


def _find_optimal_threshold(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float | None:
    """Find the decision threshold that maximises binary F1 on the test split.

    Sweeps 91 thresholds between 0.05 and 0.95 and returns the one that
    maximises F1 for the positive class (classes_[-1]).  Returns None for
    multiclass or when predict_proba is unavailable.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(classes) != 2:
            return None
        pos_label = classes[-1]
        probs = pipeline.predict_proba(X_test)[:, -1]
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 91):
            t = float(t)
            preds = pd.Series(
                [pos_label if p >= t else classes[0] for p in probs],
                index=y_test.index,
            )
            score = f1_score(y_test, preds, pos_label=pos_label, average="binary", zero_division=0)
            if score > best_f1:
                best_f1, best_thresh = score, t
        return round(best_thresh, 2)
    except Exception:
        return None


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

    # Log-transform heavily skewed non-negative regression targets
    log_transform_target = False
    if task_type == "regression" and _should_log_transform(d[target_col]):
        log_transform_target = True
        d = d.copy()
        d[target_col] = np.log1p(d[target_col].astype(float))

    imbalance_ratio: float | None = None
    xgb_scale_pos_weight: float | None = None
    if task_type == "classification":
        counts = d[target_col].value_counts()
        if len(counts) >= 2 and counts.min() > 0:
            imbalance_ratio = round(float(counts.max() / counts.min()), 2)
            if len(counts) == 2:
                xgb_scale_pos_weight = float(counts.max() / counts.min())

    # Feature type routing must happen before model selection so auto-select
    # can build comparison pipelines with the same preprocessing.
    numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols, dropped_cols = split_feature_types(d, feature_cols)

    # Lag / rolling feature engineering: triggered when the dataset has a
    # datetime sort column and the task is regression.  We sort by date,
    # create lag-1, lag-7, and rolling-mean-7 features for up to 8 numeric
    # columns, drop the first max_lag rows (insufficient history), and use a
    # temporal train/test split to prevent look-ahead leakage.
    lag_feature_cols: list[str] = []
    lag_config: dict | None = None
    temporal_split = False
    if datetime_cols and task_type == "regression" and len(d) >= 50 and numeric_cols:
        _lag_candidates = numeric_cols[:8]
        try:
            d, lag_feature_cols = engineer_lag_features(
                d, datetime_cols[0], _lag_candidates,
                lags=LAG_DEFAULTS, windows=ROLLING_DEFAULTS,
            )
            numeric_cols = numeric_cols + lag_feature_cols
            lag_config = {
                "sort_col": datetime_cols[0],
                "lag_cols": _lag_candidates,
                "lags": LAG_DEFAULTS,
                "windows": ROLLING_DEFAULTS,
            }
            temporal_split = True
        except Exception:
            lag_feature_cols = []

    usable_features = numeric_cols + categorical_cols + ordinal_cols + datetime_cols + text_cols
    if not usable_features:
        return {"error": "No usable feature columns after excluding high-cardinality/unsupported columns."}

    # Scan for features suspiciously correlated with the target before any
    # splitting — alerts on potential leakage without blocking training.
    leakage_warnings = detect_leakage(d, usable_features, target_col)

    X = d[usable_features]
    y = d[target_col]

    # Enable degree-2 pairwise interaction features when there are enough
    # numeric columns and rows to make interactions informative without
    # overfitting on tiny datasets.
    add_interactions = len(numeric_cols) >= 3 and len(d) >= 200

    stratify = None
    if task_type == "classification" and y.nunique(dropna=True) >= 2 and y.value_counts().min() >= 2:
        stratify = y

    if temporal_split:
        n = len(X)
        split_idx = int(n * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Baselines: fit before the main model so we can compare without extra splits.
    baseline_comparison: dict | None = None
    try:
        baseline_comparison = compute_baselines(y_train, y_test, task_type, log_transform_target)
    except Exception:
        pass

    # Capture training-set statistics for drift detection at scoring time.
    training_stats: dict | None = None
    try:
        training_stats = compute_training_stats(
            X_train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols + ordinal_cols,
        )
    except Exception:
        pass

    # Auto model selection: 3-candidate CV shootout on a sample.
    # Text columns are excluded from the comparison to avoid embedding the
    # training set 9× (3 candidates × 3 folds); the winning model family is
    # robust enough to the addition of extra features that this approximation
    # holds well in practice.
    auto_selection_note: str | None = None
    resolved_model_type = _resolve_model_type(task_type, model_type)
    if resolved_model_type == "auto" and len(X_train) >= 30:
        resolved_model_type, auto_selection_note = _auto_select_model(
            X_train, y_train, task_type,
            numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
            scale_pos_weight=xgb_scale_pos_weight,
        )
    elif resolved_model_type == "auto":
        # Too few rows for a meaningful comparison — use safe defaults.
        resolved_model_type = "logistic_regression" if task_type == "classification" else "ridge_regression"

    try:
        estimator = _build_estimator(task_type, resolved_model_type, scale_pos_weight=xgb_scale_pos_weight)
    except (ValueError, ImportError) as exc:
        return {"error": str(exc)}

    preprocessor = build_preprocessor(
        numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
        add_interactions=add_interactions,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    best_params: dict | None = None
    param_grid = _PARAM_GRIDS.get(resolved_model_type) if tune else None
    if param_grid and len(X_train) >= 50:
        scoring = "f1_weighted" if task_type == "classification" else "neg_mean_absolute_percentage_error"
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=HPARAM_N_ITER, cv=HPARAM_CV,
            scoring=scoring, random_state=42, n_jobs=_N_JOBS, refit=True,
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
            ("preprocess", build_preprocessor(
                numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
                add_interactions=add_interactions,
            )),
            ("model", _build_estimator(task_type, resolved_model_type, scale_pos_weight=xgb_scale_pos_weight)),
        ])
        # Temporal datasets must use TimeSeriesSplit to prevent look-ahead leakage
        # in CV fold assignment. Regular k-fold shuffles the data and contaminates
        # folds with future observations.
        cv_strategy = TimeSeriesSplit(n_splits=cv_folds) if temporal_split else cv_folds
        try:
            scores = cross_val_score(cv_pipeline, X, y, cv=cv_strategy, scoring=scoring, n_jobs=_N_JOBS)
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
    optimal_threshold: float | None = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        # CalibratedClassifierCV exposes .classes_ directly; plain Pipeline requires
        # looking inside named_steps — check both to handle both cases.
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(classes) == 2:
            probs = pipeline.predict_proba(X_test)[:, -1]
            eval_df["probability"] = probs
            probability_col = "probability"
            optimal_threshold = _find_optimal_threshold(pipeline, X_test, y_test)
            if optimal_threshold is not None and optimal_threshold != 0.5:
                eval_df["prediction"] = np.where(probs >= optimal_threshold, classes[-1], classes[0])

    if task_type == "classification":
        evaluation = evaluate_classification(
            eval_df, actual_col="actual", prediction_col="prediction", probability_col=probability_col
        )
    else:
        evaluation = evaluate_regression_or_forecast(eval_df, actual_col="actual", prediction_col="prediction")

    # Split-conformal prediction interval halfwidth for regression models.
    # Nonconformity scores = |y_actual - y_pred| on the test set.
    # The 90th-percentile score gives distribution-free 90% coverage on new data.
    conformal_halfwidth: float | None = None
    if task_type == "regression" and len(y_test_eval) >= 10:
        try:
            nonconformity = np.abs(y_test_eval - y_pred)
            n_cal = len(nonconformity)
            coverage = 0.90
            q_level = min(1.0, np.ceil((n_cal + 1) * coverage) / n_cal)
            conformal_halfwidth = float(np.quantile(nonconformity, q_level))
        except Exception:
            pass

    # Attach model metric to baseline comparison now that evaluation is ready.
    if baseline_comparison is not None:
        try:
            primary = "accuracy" if task_type == "classification" else "wmape"
            baseline_comparison = finalise_baseline_comparison(
                baseline_comparison, evaluation.get(primary)
            )
        except Exception:
            pass

    # For CalibratedClassifierCV the underlying fitted pipeline is in .estimator
    fi_source = pipeline.estimator if calibrated else pipeline
    feature_importance = _extract_feature_importance(fi_source) if hasattr(fi_source, "named_steps") else []

    import uuid as _uuid
    manager = model_manager or ModelManager()
    previous = manager.find_previous(dataset_id, target_col)
    assigned_model_id = str(_uuid.uuid4())

    # Attempt ONNX export for pipelines with only standard sklearn transformers.
    # Falls back to None silently when custom transformers are present or
    # skl2onnx/onnxruntime are not installed.  Uses the non-calibrated source
    # pipeline because CalibratedClassifierCV wraps the pipeline in a way
    # skl2onnx cannot currently handle.
    onnx_path = try_export_onnx(
        fi_source,
        X_test.head(5),
        model_id=assigned_model_id,
        model_dir=manager.model_dir,
    )

    meta = manager.save_model(
        pipeline,
        model_id=assigned_model_id,
        task_type=task_type,
        model_type=resolved_model_type,
        target_col=target_col,
        feature_cols=usable_features,
        dataset_id=dataset_id,
        log_transform_target=log_transform_target,
        evaluation=evaluation,
        optimal_threshold=optimal_threshold,
        lag_config=lag_config,
        onnx_path=onnx_path,
        training_stats=training_stats,
        conformal_halfwidth=conformal_halfwidth,
    )

    # Log every training run to the experiment tracker for later comparison.
    try:
        get_tracker().log_run(
            model_id=meta.model_id,
            dataset_id=dataset_id,
            target_col=target_col,
            task_type=task_type,
            model_type=resolved_model_type,
            params={
                "feature_cols": usable_features,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "tune": tune,
                "lag_config": lag_config,
                "add_interactions": add_interactions,
                "log_transform_target": log_transform_target,
                "best_params": best_params,
            },
            metrics={
                **evaluation,
                "cv_mean": cv_result.get("mean") if cv_result else None,
                "cv_std": cv_result.get("std") if cv_result else None,
                "optimal_threshold": optimal_threshold,
                "imbalance_ratio": imbalance_ratio,
                "calibrated": calibrated,
            },
            preprocessing={
                "dropped_cols": dropped_cols,
                "auto_dropped_id_cols": auto_dropped_id_cols,
                "datetime_feature_cols": datetime_cols,
                "lag_feature_cols": lag_feature_cols,
                "text_feature_cols": text_cols,
                "interaction_features_added": add_interactions,
            },
            comparison=None,  # filled in below after comparison is computed
        )
    except Exception:
        pass  # Experiment logging must never break training

    # Check whether the text encoder was silently skipped (embedder unavailable).
    _text_encoder_skipped = False
    _text_skip_reason = ""
    if text_cols:
        try:
            _fitted_pre = pipeline.named_steps.get("preprocess") if hasattr(pipeline, "named_steps") else None
            if _fitted_pre is None and hasattr(pipeline, "estimator"):
                _fitted_pre = pipeline.estimator.named_steps.get("preprocess")
            if _fitted_pre is not None:
                for _tname, _t, _ in getattr(_fitted_pre, "transformers_", []):
                    if _tname == "text" and getattr(_t, "_skipped", False):
                        _text_encoder_skipped = True
                        _text_skip_reason = getattr(_t, "_skip_reason", "embedder unavailable")
        except Exception:
            pass

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
    if datetime_cols:
        preprocessing_notes.append(
            f"Datetime column(s) {datetime_cols} decomposed into calendar features "
            "(year, month, day, dayofweek, is_weekend)."
        )
    if text_cols:
        if _text_encoder_skipped:
            preprocessing_notes.append(
                f"Text column(s) {text_cols} could not be embedded (embedder unavailable: "
                f"{_text_skip_reason[:120]}). These columns were dropped from the feature set."
            )
        else:
            preprocessing_notes.append(
                f"Text column(s) {text_cols} encoded as {TEXT_EMBEDDING_N_COMPONENTS}-dim sentence embeddings "
                "(all-MiniLM-L6-v2 → TruncatedSVD)."
            )
    if lag_feature_cols:
        assert lag_config is not None
        preprocessing_notes.append(
            f"Created {len(lag_feature_cols)} lag/rolling features sorted by '{lag_config['sort_col']}' "
            f"(lags: {lag_config['lags']}, rolling windows: {lag_config['windows']}). "
            "Used temporal train/test split to prevent look-ahead leakage."
        )
    if add_interactions:
        preprocessing_notes.append(
            "Degree-2 pairwise interaction features added for top numeric columns "
            "(variance-selected, pruned by VarianceThreshold, capped at 50 features)."
        )
    high_leakage = [w for w in leakage_warnings if w["risk"] == "high"]
    if high_leakage:
        cols = ", ".join(f"'{w['feature']}' (r={w['correlation']})" for w in high_leakage)
        preprocessing_notes.append(
            f"Leakage risk: {cols} — these features have very high correlation with the target. "
            "Verify they are not derivatives or proxies of the target variable."
        )
    if auto_selection_note:
        preprocessing_notes.append(auto_selection_note)

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
        "datetime_feature_cols": datetime_cols,
        "lag_feature_cols": lag_feature_cols,
        "text_feature_cols": text_cols,
        "interaction_features_added": add_interactions,
        "onnx_exported": onnx_path is not None,
        "baseline_comparison": baseline_comparison,
        "leakage_warnings": leakage_warnings,
        "conformal_halfwidth": conformal_halfwidth,
        "prediction_interval_coverage": 0.90 if conformal_halfwidth is not None else None,
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
        "optimal_threshold": optimal_threshold,
        "feature_importance": feature_importance,
        "model_comparison": model_comparison,
        "engineering_readout": _readout(
            task_type, resolved_model_type, len(X_train), len(X_test), meta.model_id, evaluation
        ) + comparison_note,
    }
