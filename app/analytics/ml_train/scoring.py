from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.analytics.ml_train.drift import check_drift, compare_fingerprints
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import engineer_lag_features
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── ONNX session cache ────────────────────────────────────────────────────────
# Loading an InferenceSession is expensive (parses the protobuf, allocates
# memory). Cache by path + mtime so repeated scoring of the same model is fast.

_ONNX_CACHE_MAX = 8
_onnx_cache: dict[str, tuple[Any, float]] = {}  # path → (session, mtime)
_onnx_cache_lock = __import__("threading").Lock()


def _load_onnx_session_cached(path: str) -> Any:
    """Return a cached onnxruntime.InferenceSession, reloading only when file changes."""
    import onnxruntime as rt
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return rt.InferenceSession(path)

    with _onnx_cache_lock:
        entry = _onnx_cache.get(path)
        if entry is not None and entry[1] == mtime:
            return entry[0]

    sess = rt.InferenceSession(path)

    with _onnx_cache_lock:
        if len(_onnx_cache) >= _ONNX_CACHE_MAX:
            oldest = next(iter(_onnx_cache))
            del _onnx_cache[oldest]
        _onnx_cache[path] = (sess, mtime)

    return sess


def _try_onnx_predict(meta, X: pd.DataFrame) -> np.ndarray | None:
    """Run ONNX inference when an exported model file exists, else return None."""
    onnx_path_str = getattr(meta, "onnx_path", None)
    if not onnx_path_str:
        return None
    onnx_path = Path(onnx_path_str)
    if not onnx_path.exists():
        return None
    try:
        sess = _load_onnx_session_cached(str(onnx_path))
        input_names = {inp.name for inp in sess.get_inputs()}
        feed: dict = {}
        for col in X.columns:
            if col not in input_names:
                continue
            arr = X[[col]].to_numpy()
            dtype = X[col].dtype
            if pd.api.types.is_float_dtype(dtype):
                feed[col] = arr.astype(np.float32)
            elif pd.api.types.is_integer_dtype(dtype):
                feed[col] = arr.astype(np.int64)
            else:
                feed[col] = arr.astype(str)
        output = sess.run(None, feed)
        preds = np.array(output[0]).ravel()
        logger.debug("ONNX inference succeeded for model %s (%d rows)", meta.model_id if hasattr(meta, 'model_id') else '?', len(preds))
        return preds
    except Exception as exc:
        logger.debug("ONNX inference failed, falling back to joblib: %s", exc)
        return None


def validate_scoring_schema(
    df: pd.DataFrame,
    model_id: str,
    model_manager: ModelManager | None = None,
) -> dict:
    """Pre-flight schema check: returns a structured diff without running scoring.

    Checks for missing features, extra columns, and type mismatches (model
    expects numeric but file has object dtype). Also runs drift and lineage
    checks when training stats are available.
    """
    manager = model_manager or ModelManager()
    try:
        _, meta = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    expected_set = set(meta.feature_cols)
    actual_set = set(df.columns)
    missing_cols = sorted(expected_set - actual_set)
    extra_cols = sorted(actual_set - expected_set - {meta.target_col})

    training_stats: dict = getattr(meta, "training_stats", None) or {}
    type_mismatches: list[dict] = []
    for col in meta.feature_cols:
        if col not in df.columns:
            continue
        stat = training_stats.get(col)
        if stat and stat.get("type") == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                type_mismatches.append({
                    "feature": col,
                    "expected_type": "numeric",
                    "actual_dtype": str(df[col].dtype),
                })

    avail = [c for c in meta.feature_cols if c in df.columns]
    drift_report: dict | None = None
    if training_stats and avail:
        try:
            drift_report = check_drift(df[avail], training_stats)
        except Exception:
            pass

    lineage_report: dict | None = None
    stored_fp = getattr(meta, "data_fingerprint", None)
    if stored_fp and avail:
        try:
            lineage_report = compare_fingerprints(df[avail], avail, stored_fp)
        except Exception:
            pass

    schema_ok = not missing_cols and not type_mismatches
    return {
        "model_id": model_id,
        "target_col": meta.target_col,
        "task_type": meta.task_type,
        "n_rows": int(len(df)),
        "n_expected_features": len(meta.feature_cols),
        "schema_ok": schema_ok,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "type_mismatches": type_mismatches,
        "drift": drift_report,
        "lineage": lineage_report,
    }


def score_with_model(
    df: pd.DataFrame,
    model_id: str,
    top_n: int = 500,
    model_manager: ModelManager | None = None,
) -> dict:
    _t0 = time.perf_counter()
    manager = model_manager or ModelManager()
    pipeline, meta = manager.load_model(model_id)

    # Re-apply lag/rolling features when the model was trained with them.
    lag_config = getattr(meta, "lag_config", None)
    if lag_config:
        df, _ = engineer_lag_features(
            df,
            lag_config["sort_col"],
            lag_config["lag_cols"],
            lags=lag_config.get("lags", [1, 7]),
            windows=lag_config.get("windows", [7]),
        )

    missing = [c for c in meta.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns required by model {model_id}: {missing}")

    X = df[meta.feature_cols]
    predictions = _try_onnx_predict(meta, X)
    using_onnx = predictions is not None
    if predictions is None:
        predictions = pipeline.predict(X)

    if getattr(meta, "log_transform_target", False):
        predictions = np.expm1(predictions)

    out = df.copy()
    out["prediction"] = predictions

    # Regression: conformal prediction intervals
    halfwidth = getattr(meta, "conformal_halfwidth", None)
    if halfwidth is not None and meta.task_type == "regression":
        lower = predictions - halfwidth
        if getattr(meta, "log_transform_target", False):
            lower = lower.clip(min=0)
        out["prediction_lower_90"] = lower
        out["prediction_upper_90"] = predictions + halfwidth

    # Classification: probabilities + conformal prediction sets
    conf_clf_threshold = getattr(meta, "conformal_classification_threshold", None)
    pred_set_info: dict | None = None
    if meta.task_type == "classification" and hasattr(pipeline, "predict_proba"):
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))

        if len(classes) >= 2:
            probs = pipeline.predict_proba(X)

            # Binary probability column
            if len(classes) == 2:
                prob_col = probs[:, -1]
                out["prediction_probability"] = prob_col
                threshold = getattr(meta, "optimal_threshold", None) or 0.5
                if threshold != 0.5:
                    out["prediction"] = np.where(prob_col >= threshold, classes[-1], classes[0])

            # Conformal prediction sets
            if conf_clf_threshold is not None:
                pred_sets = []
                for row_probs in probs:
                    ps = [str(classes[i]) for i, p in enumerate(row_probs) if p >= 1.0 - conf_clf_threshold]
                    pred_sets.append(
                        "|".join(sorted(ps)) if ps else str(classes[int(np.argmax(row_probs))])
                    )
                out["prediction_set"] = pred_sets
                avg_set_size = float(np.mean([len(s.split("|")) for s in pred_sets]))
                pred_set_info = {
                    "coverage_target": 0.90,
                    "threshold": round(conf_clf_threshold, 4),
                    "avg_set_size": round(avg_set_size, 2),
                    "n_singleton": int(sum(1 for s in pred_sets if "|" not in s)),
                }

    n_rows = len(out)
    scored_rows = out.head(top_n).reset_index(drop=True).to_dict(orient="records")

    drift_report: dict | None = None
    training_stats = getattr(meta, "training_stats", None)
    if training_stats:
        try:
            drift_report = check_drift(X, training_stats)
        except Exception:
            pass

    # Data lineage: compare scoring column set and distributions to training fingerprint
    lineage_report: dict | None = None
    stored_fp = getattr(meta, "data_fingerprint", None)
    if stored_fp:
        try:
            lineage_report = compare_fingerprints(X, meta.feature_cols, stored_fp)
        except Exception:
            pass

    pi_note = (
        f"90% prediction intervals included (±{halfwidth:.4f})."
        if halfwidth is not None and meta.task_type == "regression"
        else None
    )
    ps_note = (
        f"Conformal prediction sets (90% coverage) added as prediction_set column — avg size {pred_set_info['avg_set_size']:.1f}."
        if pred_set_info
        else None
    )

    latency_ms = round((time.perf_counter() - _t0) * 1000, 1)

    # Record per-model scoring latency for the health endpoint.
    try:
        from app.agent.latency_metrics import scoring_latency
        scoring_latency.record(model_id, latency_ms)
    except Exception:
        pass

    # Auto-retrain when high drift detected and the feature is enabled.
    auto_retrain_job_id: str | None = None
    if (
        drift_report
        and drift_report.get("overall_severity") == "high"
        and settings.auto_retrain_on_high_drift
        and meta.dataset_id
    ):
        try:
            from app.api.training_jobs import submit_job
            from app.analytics.ml_train.auto_retrain import auto_retrain_model
            auto_retrain_job_id = submit_job(
                auto_retrain_model,
                model_id,
                meta.dataset_id,
                meta.target_col,
                settings.auto_retrain_model_type,
            )
        except Exception:
            pass

    return {
        "model_id": model_id,
        "dataset_id": meta.dataset_id,
        "task_type": meta.task_type,
        "target_col": meta.target_col,
        "n_rows_scored": n_rows,
        "scored_rows": scored_rows,
        "drift": drift_report,
        "lineage": lineage_report,
        "prediction_set_info": pred_set_info,
        "conformal_halfwidth": halfwidth,
        "scoring_latency_ms": latency_ms,
        "auto_retrain_job_id": auto_retrain_job_id,
        "using_onnx": using_onnx,
        "engineering_readout": (
            f"Scored {n_rows} rows with model {model_id} ({meta.model_type}, {meta.task_type}"
            + (", ONNX runtime" if using_onnx else "")
            + ")."
            + (f" {pi_note}" if pi_note else "")
            + (f" {ps_note}" if ps_note else "")
            + (f" Auto-retrain queued (job {auto_retrain_job_id[:8]})." if auto_retrain_job_id else "")
        ),
    }
