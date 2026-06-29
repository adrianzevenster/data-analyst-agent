from __future__ import annotations

import os
from typing import Any, Callable, Literal

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.analytics.ml_train.preprocessing import build_preprocessor

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

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _SMOTE_AVAILABLE = True
except ImportError:
    SMOTE = None  # type: ignore[assignment,misc]
    ImbPipeline = None  # type: ignore[assignment,misc]
    _SMOTE_AVAILABLE = False

# Classifiers that lack class_weight support — SMOTE is the primary remedy.
_NO_CLASS_WEIGHT_CLASSIFIERS = frozenset({
    "gradient_boosting_classifier",
    "decision_tree_classifier",
    "knn_classifier",
})
# Models that handle imbalance natively (scale_pos_weight / class_weight) — only
# use SMOTE on top when imbalance is very severe (ratio > 10).
_NATIVE_IMBALANCE_CLASSIFIERS = frozenset({
    "xgboost_classifier",
    "lightgbm_classifier",
    "logistic_regression",
    "random_forest_classifier",
})

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
    # Generic family names — resolved to a task-specific variant at runtime.
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "knn",
    "xgboost",
    "lightgbm",
]

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

# Candidates for auto model selection per task type.  Built dynamically so the
# shootout always includes every gradient-boosting library that is installed.
def _build_auto_candidates() -> dict[str, list[str]]:
    clf = ["logistic_regression", "random_forest_classifier"]
    reg = ["ridge_regression", "random_forest_regressor"]
    if XGBClassifier is not None:
        clf.append("xgboost_classifier")
        reg.append("xgboost_regressor")
    if LGBMClassifier is not None:
        clf.append("lightgbm_classifier")
        reg.append("lightgbm_regressor")
    if XGBClassifier is None and LGBMClassifier is None:
        clf.append("gradient_boosting_classifier")
        reg.append("gradient_boosting_regressor")
    return {"classification": clf, "regression": reg}


_AUTO_CANDIDATES = _build_auto_candidates()
# Max rows per candidate during auto-select — relative ranking doesn't need full data.
_AUTO_SELECT_SAMPLE = 2000


def _require_installed(cls: Any, package_name: str) -> Any:
    if cls is None:
        raise ImportError(
            f"{package_name} is not installed in this environment. "
            f"Add it to requirements-api.txt to use this model_type."
        )
    return cls


# Builders are lazy (lambdas) so an unavailable optional dependency only raises
# when that specific model_type is requested, not at import time.
CLASSIFIER_BUILDERS: dict[str, Callable[[], Any]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest_classifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=_N_JOBS, class_weight="balanced"),
    "gradient_boosting_classifier": lambda: GradientBoostingClassifier(random_state=42),
    "decision_tree_classifier": lambda: DecisionTreeClassifier(random_state=42, max_depth=8, class_weight="balanced"),
    "knn_classifier": lambda: KNeighborsClassifier(),
    "xgboost_classifier": lambda: _require_installed(XGBClassifier, "xgboost")(
        eval_metric="logloss", random_state=42
    ),
    "lightgbm_classifier": lambda: _require_installed(LGBMClassifier, "lightgbm")(
        random_state=42, verbosity=-1, class_weight="balanced"
    ),
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


def _build_estimator(task_type: str, model_type: str, scale_pos_weight: float | None = None) -> Any:
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


def _resolve_model_type(task_type: str, model_type: str) -> str:
    if model_type == "auto":
        return "auto"
    if model_type in MODEL_FAMILY_ALIASES:
        return MODEL_FAMILY_ALIASES[model_type][task_type]
    return model_type


def should_use_smote(
    task_type: str,
    resolved_model_type: str,
    imbalance_ratio: float | None,
    y_train: pd.Series,
) -> tuple[bool, int]:
    """Return (use_smote, k_neighbors)."""
    if not _SMOTE_AVAILABLE or task_type != "classification":
        return False, 5
    if imbalance_ratio is None or y_train.nunique() != 2:
        return False, 5
    minority_count = int(y_train.value_counts().min())
    if minority_count < 6:
        return False, 5
    use = (
        resolved_model_type in _NO_CLASS_WEIGHT_CLASSIFIERS and imbalance_ratio > 5
    ) or imbalance_ratio > 10
    return use, min(5, minority_count - 1)


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
    """N-candidate CV shootout on a capped sample. Returns (best_model_type, note)."""
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
    n_cands = len(candidates)
    note = f"Auto-selected {best_type} via {n_cands}-candidate 3-fold CV ({scoring}: {score_summary})"
    return best_type, note
