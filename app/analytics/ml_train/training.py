from __future__ import annotations

import uuid as _uuid

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.analytics.ml_eval.classification import evaluate_classification
from app.analytics.ml_eval.regression import evaluate_regression_or_forecast
from app.analytics.ml_train.baseline import compute_baselines, finalise_baseline_comparison
from app.analytics.ml_train.drift import compute_fingerprint, compute_training_stats
from app.analytics.ml_train.experiment_tracker import get_tracker
from app.analytics.ml_train.fit_postprocess import (
    _extract_feature_importance,
    _find_optimal_threshold,
    compute_conformal_classification,
    compute_conformal_regression,
    compute_ovr_roc,
)
from app.analytics.ml_train.leakage import detect_leakage
from app.analytics.ml_train.model_catalog import (
    HPARAM_CV,
    HPARAM_N_ITER,
    ImbPipeline,
    LGBMClassifier,
    ModelType,
    SMOTE,
    TaskHint,
    XGBClassifier,
    _AUTO_CANDIDATES,
    _CALIBRATION_CLASSIFIERS,
    _N_JOBS,
    _PARAM_GRIDS,
    _SMOTE_AVAILABLE,
    _auto_select_model,
    _build_estimator,
    _resolve_model_type,
    should_use_smote,
)
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.onnx_export import try_export_onnx
from app.analytics.ml_train.preprocessing import (
    LAG_DEFAULTS,
    ROLLING_DEFAULTS,
    TEXT_EMBEDDING_N_COMPONENTS,
    build_preprocessor,
    engineer_lag_features,
    select_features,
    split_feature_types,
)
from app.analytics.ml_train.task_utils import (
    _infer_task_type,
    _looks_like_id_col,
    _readout,
    _should_log_transform,
)

# Re-export symbols that external modules (tests, tooling) import from this module.
__all__ = [
    "train_supervised_model",
    "TaskHint",
    "ModelType",
    "_AUTO_CANDIDATES",
    "_SMOTE_AVAILABLE",
    "LGBMClassifier",
    "XGBClassifier",
]


def train_supervised_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_hint: TaskHint = "auto",
    model_type: ModelType = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune: bool = True,
    max_rows: int | None = None,
    dataset_id: str | None = None,
    model_manager: ModelManager | None = None,
) -> dict:
    feature_cols = [c for c in (feature_cols or df.columns) if c != target_col]

    d = df[[target_col] + feature_cols].dropna(subset=[target_col])
    if d.empty:
        return {"error": "No non-null rows for the target column."}

    source_rows = int(len(d))
    sampled_rows = False
    if max_rows is not None and source_rows > max_rows:
        d = d.sample(n=max_rows, random_state=42).sort_index()
        sampled_rows = True

    # Drop identifier / sequential columns — they cause spurious correlations.
    auto_dropped_id_cols = [c for c in feature_cols if _looks_like_id_col(d, c)]
    if auto_dropped_id_cols:
        feature_cols = [c for c in feature_cols if c not in set(auto_dropped_id_cols)]

    task_type = _infer_task_type(d[target_col], task_hint)

    # Log-transform heavily skewed non-negative regression targets.
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
    # datetime sort column and the task is regression.
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

    usable_features, _feat_sel_notes = select_features(d, usable_features)
    if not usable_features:
        return {"error": "No usable features remain after feature selection (all near-constant or correlated)."}

    # Re-partition after selection so the preprocessor only sees selected features.
    _sel = set(usable_features)
    numeric_cols = [c for c in numeric_cols if c in _sel]
    categorical_cols = [c for c in categorical_cols if c in _sel]
    ordinal_cols = [c for c in ordinal_cols if c in _sel]
    datetime_cols = [c for c in datetime_cols if c in _sel]
    text_cols = [c for c in text_cols if c in _sel]

    # Scan for features suspiciously correlated with the target before splitting.
    leakage_warnings = detect_leakage(d, usable_features, target_col)

    X = d[usable_features]
    y = d[target_col]

    add_interactions = len(numeric_cols) >= 3 and len(d) >= 200

    stratify = None
    if task_type == "classification" and y.nunique(dropna=True) >= 2 and y.value_counts().min() >= 2:
        stratify = y

    if temporal_split:
        n = len(X)
        split_idx = int(n * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    baseline_comparison: dict | None = None
    try:
        baseline_comparison = compute_baselines(y_train, y_test, task_type, log_transform_target)
    except Exception:
        pass

    training_stats: dict | None = None
    try:
        training_stats = compute_training_stats(
            X_train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols + ordinal_cols,
        )
    except Exception:
        pass

    # Auto model selection: N-candidate CV shootout on a sample.
    # Text columns are excluded from the comparison to avoid embedding the
    # training set N×3 times; the winning model family is robust enough to
    # the addition of extra features.
    auto_selection_note: str | None = None
    resolved_model_type = _resolve_model_type(task_type, model_type)
    if resolved_model_type == "auto" and len(X_train) >= 30:
        resolved_model_type, auto_selection_note = _auto_select_model(
            X_train, y_train, task_type,
            numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
            scale_pos_weight=xgb_scale_pos_weight,
        )
    elif resolved_model_type == "auto":
        resolved_model_type = "logistic_regression" if task_type == "classification" else "ridge_regression"

    _use_smote, _smote_k = should_use_smote(task_type, resolved_model_type, imbalance_ratio, y_train)

    try:
        estimator = _build_estimator(task_type, resolved_model_type, scale_pos_weight=xgb_scale_pos_weight)
    except (ValueError, ImportError) as exc:
        return {"error": str(exc)}

    preprocessor = build_preprocessor(
        numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
        add_interactions=add_interactions,
    )
    _steps = [("preprocess", preprocessor)]
    if _use_smote and SMOTE is not None and ImbPipeline is not None:
        _steps.append(("smote", SMOTE(random_state=42, k_neighbors=_smote_k)))
        pipeline = ImbPipeline(_steps + [("model", estimator)])
    else:
        pipeline = Pipeline(_steps + [("model", estimator)])

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
        _cv_steps = [
            ("preprocess", build_preprocessor(
                numeric_cols, categorical_cols, ordinal_cols, datetime_cols, text_cols,
                add_interactions=add_interactions,
            )),
        ]
        if _use_smote and SMOTE is not None and ImbPipeline is not None:
            _cv_steps.append(("smote", SMOTE(random_state=42, k_neighbors=_smote_k)))
            cv_pipeline = ImbPipeline(_cv_steps + [
                ("model", _build_estimator(task_type, resolved_model_type, scale_pos_weight=xgb_scale_pos_weight)),
            ])
        else:
            cv_pipeline = Pipeline(_cv_steps + [
                ("model", _build_estimator(task_type, resolved_model_type, scale_pos_weight=xgb_scale_pos_weight)),
            ])
        # Temporal datasets must use TimeSeriesSplit to prevent look-ahead leakage.
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
        # Return to original scale for evaluation so WMAPE/R² are interpretable.
        y_pred = np.expm1(y_pred)
        y_test_eval = np.expm1(y_test.to_numpy())
    else:
        y_test_eval = y_test.to_numpy()
    eval_df = pd.DataFrame({"actual": y_test_eval, "prediction": y_pred})

    probability_col = None
    optimal_threshold: float | None = None
    _pipeline_classes: list = []
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        # CalibratedClassifierCV exposes .classes_ directly; plain Pipeline requires
        # looking inside named_steps — check both to handle both cases.
        _cls = getattr(pipeline, "classes_", None)
        _pipeline_classes = list(_cls) if _cls is not None else []
        if not _pipeline_classes and hasattr(pipeline, "named_steps"):
            _pipeline_classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(_pipeline_classes) == 2:
            probs = pipeline.predict_proba(X_test)[:, -1]
            eval_df["probability"] = probs
            probability_col = "probability"
            optimal_threshold = _find_optimal_threshold(pipeline, X_test, y_test)
            if optimal_threshold is not None and optimal_threshold != 0.5:
                eval_df["prediction"] = np.where(probs >= optimal_threshold, _pipeline_classes[-1], _pipeline_classes[0])

    if task_type == "classification":
        evaluation = evaluate_classification(
            eval_df, actual_col="actual", prediction_col="prediction", probability_col=probability_col
        )
    else:
        evaluation = evaluate_regression_or_forecast(eval_df, actual_col="actual", prediction_col="prediction")

    # OVR ROC curves for multiclass classifiers.
    if task_type == "classification" and len(_pipeline_classes) > 2 and hasattr(pipeline, "predict_proba"):
        evaluation.update(compute_ovr_roc(pipeline, X_test, y_test, _pipeline_classes))

    conformal_halfwidth = compute_conformal_regression(y_test_eval, y_pred) if task_type == "regression" else None
    conformal_classification_threshold = (
        compute_conformal_classification(pipeline, X_test, y_test)
        if task_type == "classification"
        else None
    )

    data_fingerprint: dict | None = None
    try:
        data_fingerprint = compute_fingerprint(X_train, usable_features)
    except Exception:
        pass

    if baseline_comparison is not None:
        try:
            primary = "accuracy" if task_type == "classification" else "wmape"
            baseline_comparison = finalise_baseline_comparison(
                baseline_comparison, evaluation.get(primary)
            )
        except Exception:
            pass

    # For CalibratedClassifierCV the underlying fitted pipeline is in .estimator.
    fi_source = pipeline.estimator if calibrated else pipeline
    feature_importance = _extract_feature_importance(fi_source) if hasattr(fi_source, "named_steps") else []

    manager = model_manager or ModelManager()
    previous = manager.find_previous(dataset_id, target_col)
    assigned_model_id = str(_uuid.uuid4())

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
        conformal_classification_threshold=conformal_classification_threshold,
        data_fingerprint=data_fingerprint,
    )

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
                "max_rows": max_rows,
                "source_rows": source_rows,
                "sampled_rows": sampled_rows,
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
            comparison=None,
        )
    except Exception:
        pass  # Experiment logging must never break training.

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

    preprocessing_notes: list[str] = list(_feat_sel_notes)
    if _use_smote:
        preprocessing_notes.append(
            f"SMOTE applied (imbalance ratio {imbalance_ratio}×): synthetic minority samples generated "
            f"during training (k_neighbors={_smote_k}). Evaluation metrics reflect the original distribution."
        )
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
            metric_name = "accuracy"
            improved = bool((curr_v or 0.0) > (prev_v or 0.0))
        else:
            prev_v = prev_eval.get("wmape")
            curr_v = evaluation.get("wmape")
            metric_name = "wmape"
            improved = bool((curr_v or float("inf")) < (prev_v or float("inf")))

        if prev_v is not None and curr_v is not None:
            delta = round(float(curr_v) - float(prev_v), 4)
            model_comparison = {
                "previous_model_id": previous.model_id,
                "previous_model_type": previous.model_type,
                "metric": metric_name,
                "previous": round(float(prev_v), 4),
                "current": round(float(curr_v), 4),
                "delta": delta,
                "improved": improved,
            }
            arrow = "↑" if improved else "↓"
            comparison_note = (
                f" vs previous {previous.model_type}: "
                f"{metric_name.upper()} {round(float(prev_v), 4)} → {round(float(curr_v), 4)} "
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
        "n_rows_source": source_rows,
        "max_rows": max_rows,
        "sampled_rows": sampled_rows,
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
