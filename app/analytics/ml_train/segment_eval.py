from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.model_store import ModelManager


def evaluate_by_segment(
    df: pd.DataFrame,
    model_id: str,
    segment_col: str,
    actual_col: str | None = None,
    model_manager: ModelManager | None = None,
) -> dict:
    """Evaluate model performance broken down by a categorical segment column.

    Surfaces per-segment accuracy/F1 (classification) or RMSE/WMAPE (regression),
    making fairness gaps and cohort-level performance differences visible.
    """
    manager = model_manager or ModelManager()
    try:
        meta = manager.get_meta(model_id)
        pipeline, _ = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    if segment_col not in df.columns:
        return {"error": f"Segment column '{segment_col}' not found in dataset."}

    resolved_actual = actual_col or (meta.target_col if meta.target_col in df.columns else None)
    if resolved_actual is None:
        return {
            "error": (
                f"Target column '{meta.target_col}' not in dataset. "
                "Provide actual_col pointing to the ground-truth labels."
            )
        }

    missing_feats = [c for c in meta.feature_cols if c not in df.columns]
    if missing_feats:
        return {"error": f"Dataset missing model feature(s): {', '.join(missing_feats[:5])}"}

    X = df[meta.feature_cols]
    try:
        y_pred = pipeline.predict(X)
    except Exception as exc:
        return {"error": f"Scoring failed: {exc}"}

    work = df[[segment_col]].copy()
    work["_actual"] = df[resolved_actual].values
    work["_pred"] = y_pred

    if meta.task_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        def _metrics(grp: pd.DataFrame) -> dict:
            ya, yp = grp["_actual"], grp["_pred"]
            return {
                "accuracy": round(float(accuracy_score(ya, yp)), 4),
                "f1_weighted": round(float(f1_score(ya, yp, average="weighted", zero_division=0)), 4),
            }

        primary = "accuracy"
    else:
        def _metrics(grp: pd.DataFrame) -> dict:
            ya = grp["_actual"].astype(float)
            yp = grp["_pred"].astype(float)
            rmse = float(np.sqrt(np.mean((ya - yp) ** 2)))
            wmape = float(np.abs(ya - yp).sum() / (np.abs(ya).sum() + 1e-9))
            return {"rmse": round(rmse, 4), "wmape": round(wmape, 4)}

        primary = "rmse"

    rows: list[dict] = []
    for seg_val, grp in work.groupby(segment_col, sort=True):
        if len(grp) < 2:
            continue
        m = _metrics(grp)
        rows.append({"segment": str(seg_val), "n": len(grp), **m})

    if not rows:
        return {"error": "No segments with sufficient data (need ≥ 2 rows per segment)."}

    overall = _metrics(work)
    rows.append({"segment": "__overall__", "n": len(work), **overall})

    chart_rows = [r for r in rows if r["segment"] != "__overall__"]
    chart = {
        "type": "bar",
        "title": f"{primary.upper()} by {segment_col} — model: {meta.target_col}",
        "x": "segment",
        "y": primary,
        "data": [{"segment": r["segment"], primary: r[primary]} for r in chart_rows],
    }

    seg_values = [r[primary] for r in chart_rows]
    readout = (
        f"Segmented evaluation of model {model_id[:8]} (target: '{meta.target_col}') "
        f"across {len(chart_rows)} segment(s) of '{segment_col}'. "
        f"Overall {primary}: {overall[primary]:.4f}. "
        f"Range: {min(seg_values):.4f} – {max(seg_values):.4f}."
    )

    return {
        "model_id": model_id,
        "target_col": meta.target_col,
        "task_type": meta.task_type,
        "segment_col": segment_col,
        "segments": rows,
        "charts": [chart],
        "engineering_readout": readout,
    }
