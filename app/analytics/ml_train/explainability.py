from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.analytics.ml_train.model_store import ModelManager

_TREE_MODELS = frozenset({
    "random_forest_classifier",
    "random_forest_regressor",
    "gradient_boosting_classifier",
    "gradient_boosting_regressor",
    "decision_tree_classifier",
    "decision_tree_regressor",
    "xgboost_classifier",
    "xgboost_regressor",
    "lightgbm_classifier",
    "lightgbm_regressor",
})

_LINEAR_MODELS = frozenset({
    "logistic_regression",
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
})


def _unwrap_pipeline(pipeline):
    """Strip CalibratedClassifierCV and return (preprocessor, raw_model).

    Returns (None, None) when the pipeline layout is unrecognised so callers
    can fall back to permutation importance.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(pipeline, CalibratedClassifierCV):
            pipeline = pipeline.calibrated_classifiers_[0].estimator
    except (AttributeError, IndexError):
        pass

    if hasattr(pipeline, "named_steps"):
        return (
            pipeline.named_steps.get("preprocess"),
            pipeline.named_steps.get("model"),
        )
    return None, None


def _to_dense(X) -> np.ndarray:
    import scipy.sparse
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _shap_mean_abs(sv) -> np.ndarray:
    """Reduce any SHAP output shape to per-feature mean absolute values."""
    if isinstance(sv, list):
        # Old-style multi-class: list[(n_samples, n_features)]
        if len(sv) == 2:
            return np.abs(sv[1]).mean(axis=0)
        return np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)

    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes)
        if sv.shape[-1] == 2:
            return np.abs(sv[:, :, 1]).mean(axis=0)   # binary: positive class
        return np.abs(sv).mean(axis=(0, 2))            # multiclass: mean across classes
    # (n_samples, n_features) — regression or XGBoost binary
    return np.abs(sv).mean(axis=0)


def _aggregate_text_embeddings(raw: list[dict]) -> list[dict]:
    """Sum SHAP values for text__{col}__emb_{i} features into one entry per column.

    Also strips ColumnTransformer prefixes (numeric__, categorical__, etc.) from
    all other feature names so the output is human-readable.
    """
    aggregated: dict[str, dict] = {}
    text_dims: dict[str, int] = {}

    for item in raw:
        name: str = item["feature"]
        val: float = item["shap_mean_abs"]

        if name.startswith("text__") and "__emb_" in name:
            col_name = name[len("text__"):].split("__emb_")[0]
            text_dims[col_name] = text_dims.get(col_name, 0) + 1
            key = f"_text_{col_name}"
            if key not in aggregated:
                aggregated[key] = {"feature": col_name, "shap_mean_abs": 0.0}
            aggregated[key]["shap_mean_abs"] += val
        else:
            display = name.split("__", 1)[1] if "__" in name else name
            aggregated[name] = {"feature": display, "shap_mean_abs": round(val, 6)}

    for col_name, n_dims in text_dims.items():
        key = f"_text_{col_name}"
        if key in aggregated:
            aggregated[key]["feature"] = f"{col_name} (text, {n_dims} dims)"
            aggregated[key]["shap_mean_abs"] = round(aggregated[key]["shap_mean_abs"], 6)

    return sorted(aggregated.values(), key=lambda x: -x["shap_mean_abs"])


def explain_model(
    df: pd.DataFrame,
    model_id: str,
    sample: int = 500,
    n_repeats: int = 10,
    model_manager: ModelManager | None = None,
) -> dict:
    """SHAP feature importance for a stored model evaluated on the current dataset.

    Uses TreeExplainer for tree/ensemble models (fast, exact), LinearExplainer
    for linear models, and falls back to permutation importance for KNN or on
    any unexpected SHAP error.  Text embedding columns (text__{col}__emb_{i})
    are summed back to a single per-column score for interpretability.
    """
    manager = model_manager or ModelManager()
    try:
        pipeline, meta = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Dataset missing model features: {', '.join(missing)}"}
    if meta.target_col not in df.columns:
        return {"error": f"Target column '{meta.target_col}' not found in dataset."}

    d = df[meta.feature_cols + [meta.target_col]].dropna(subset=[meta.target_col])
    if d.empty:
        return {"error": "No rows remain after dropping nulls on target column."}
    if len(d) > sample:
        d = d.sample(n=sample, random_state=42)

    X = d[meta.feature_cols]
    y = d[meta.target_col]

    if meta.log_transform_target and meta.task_type == "regression":
        y = np.log1p(pd.to_numeric(y, errors="coerce").fillna(0).astype(float))

    preprocessor, model = _unwrap_pipeline(pipeline)

    if preprocessor is None or model is None:
        return _permutation_fallback(pipeline, X, y, meta, n_repeats)

    try:
        X_arr = _to_dense(preprocessor.transform(X))
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return _permutation_fallback(pipeline, X, y, meta, n_repeats)

    model_type = meta.model_type

    try:
        import shap
    except ImportError:
        return _permutation_fallback(pipeline, X, y, meta, n_repeats)

    method = "shap_tree"
    sv = None

    if model_type in _TREE_MODELS:
        try:
            explainer = shap.TreeExplainer(model)
            exp = explainer(X_arr, check_additivity=False)
            sv = exp.values
        except Exception:
            pass

    elif model_type in _LINEAR_MODELS:
        try:
            method = "shap_linear"
            background = X_arr[: min(100, len(X_arr))]
            explainer = shap.LinearExplainer(model, background)
            exp = explainer(X_arr)
            sv = exp.values
        except Exception:
            pass

    if sv is None:
        return _permutation_fallback(pipeline, X, y, meta, n_repeats)

    mean_abs = _shap_mean_abs(sv)
    if mean_abs.shape[0] != len(feature_names):
        return _permutation_fallback(pipeline, X, y, meta, n_repeats)

    raw = sorted(
        [
            {"feature": name, "shap_mean_abs": round(float(val), 6)}
            for name, val in zip(feature_names, mean_abs)
        ],
        key=lambda x: -x["shap_mean_abs"],
    )

    aggregated = _aggregate_text_embeddings(raw)
    top = aggregated[:15]
    top_name = top[0]["feature"] if top else "n/a"
    top_val = top[0]["shap_mean_abs"] if top else 0.0

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "n_samples": int(len(d)),
        "method": method,
        "feature_importances": top,
        "raw_feature_importances": raw[:50],
        "engineering_readout": (
            f"SHAP ({method}) for {meta.model_type} predicting '{meta.target_col}' "
            f"on {len(d)} samples. Top feature: '{top_name}' (mean |SHAP|={top_val:.4f})."
        ),
    }


def _permutation_fallback(pipeline, X, y, meta, n_repeats: int) -> dict:
    """Permutation importance when SHAP cannot be applied (e.g. KNN)."""
    scoring = "f1_weighted" if meta.task_type == "classification" else "r2"
    try:
        perm = permutation_importance(
            pipeline, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )
    except Exception as exc:
        return {"error": f"Permutation importance computation failed: {exc}"}

    importances = sorted(
        [
            {
                "feature": meta.feature_cols[i],
                "shap_mean_abs": round(float(abs(perm.importances_mean[i])), 6),
                "importance_mean": round(float(perm.importances_mean[i]), 6),
                "importance_std": round(float(perm.importances_std[i]), 6),
            }
            for i in range(len(meta.feature_cols))
        ],
        key=lambda x: -x["shap_mean_abs"],
    )

    top = importances[:15]
    top_name = top[0]["feature"] if top else "n/a"
    top_val = top[0]["shap_mean_abs"] if top else 0.0
    negative_count = sum(1 for f in importances if f.get("importance_mean", 0) < 0)
    noise_note = f" {negative_count} feature(s) had negative importance (noise)." if negative_count else ""

    return {
        "model_id": meta.model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "n_samples": int(len(X)),
        "method": "permutation",
        "feature_importances": top,
        "engineering_readout": (
            f"Permutation importance ({scoring}) for {meta.model_type} predicting "
            f"'{meta.target_col}' on {len(X)} samples. "
            f"Top feature: '{top_name}' (mean |Δ|={top_val:.4f}).{noise_note}"
        ),
    }
