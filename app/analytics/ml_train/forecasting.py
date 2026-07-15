from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.preprocessing import engineer_lag_features, LAG_DEFAULTS, ROLLING_DEFAULTS


def _infer_step_offset(series: pd.Series) -> pd.DateOffset:
    """Return the modal gap between consecutive timestamps as a DateOffset."""
    try:
        diffs = series.sort_values().diff().dropna()
        if diffs.empty:
            return pd.DateOffset(days=1)
        mode_diff = diffs.mode().iloc[0]
        days = int(mode_diff.total_seconds() / 86_400)
        if days <= 1:
            return pd.DateOffset(days=1)
        if days <= 7:
            return pd.DateOffset(days=7)
        if days <= 31:
            return pd.DateOffset(months=1)
        return pd.DateOffset(years=1)
    except Exception:
        return pd.DateOffset(days=1)


def _holt_forecast(y_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Holt's linear (double exponential smoothing) in pure numpy.

    Optimises alpha and beta via grid search minimising 1-step-ahead SSE,
    then forecasts `horizon` steps ahead from the fitted level and trend.
    No statsmodels dependency required.
    """
    n = len(y_hist)
    if n < 4:
        return np.full(horizon, float(np.mean(y_hist)))

    def _run(alpha: float, beta: float) -> tuple[float, float, float]:
        lvl = float(y_hist[0])
        b = float(y_hist[1] - y_hist[0]) if n > 1 else 0.0
        sse = 0.0
        for i in range(1, n):
            one_step = lvl + b
            sse += (float(y_hist[i]) - one_step) ** 2
            lvl_new = alpha * float(y_hist[i]) + (1.0 - alpha) * one_step
            b_new = beta * (lvl_new - lvl) + (1.0 - beta) * b
            lvl, b = lvl_new, b_new
        return lvl, b, sse

    best_alpha, best_beta, best_sse = 0.3, 0.1, float("inf")
    for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        for b in (0.05, 0.1, 0.2, 0.3, 0.5):
            _, _, sse = _run(a, b)
            if sse < best_sse:
                best_alpha, best_beta, best_sse = a, b, sse

    l_final, b_final, _ = _run(best_alpha, best_beta)
    return np.array([l_final + h * b_final for h in range(1, horizon + 1)])


def _ml_rollout(
    current: pd.DataFrame,
    horizon: int,
    pipeline,
    meta,
    sort_col: str,
    lag_cols: list[str],
    lags: list[int],
    windows: list[int],
    step_offset: pd.DateOffset,
) -> list[float] | None:
    """Autoregressive rollout returning predicted target values (original scale)."""
    preds: list[float] = []
    cur = current.copy()
    next_date = cur[sort_col].iloc[-1] + step_offset

    for _ in range(horizon):
        try:
            feat_df, _ = engineer_lag_features(cur, sort_col, lag_cols, lags=lags, windows=windows)
        except Exception:
            return None

        last_feat = feat_df.iloc[[-1]]
        last_raw = cur.iloc[-1]
        X_dict: dict = {}
        for col in meta.feature_cols:
            if col in last_feat.columns:
                X_dict[col] = last_feat[col].values
            elif col in last_raw.index:
                X_dict[col] = [last_raw[col]]
            else:
                X_dict[col] = [np.nan]

        try:
            pred_raw = float(pipeline.predict(pd.DataFrame(X_dict))[0])
        except Exception:
            return None

        pred = np.expm1(pred_raw) if getattr(meta, "log_transform_target", False) else pred_raw
        preds.append(float(pred))

        new_row = cur.iloc[[-1]].copy()
        new_row[sort_col] = next_date
        new_row[meta.target_col] = pred
        for c in lag_cols:
            if c in new_row.columns and c != meta.target_col:
                new_row[c] = pred
        cur = pd.concat([cur, new_row], ignore_index=True)
        next_date = next_date + step_offset

    return preds


def _holdout_comparison(
    df_work: pd.DataFrame,
    horizon: int,
    seed_size: int,
    pipeline,
    meta,
    sort_col: str,
    lag_cols: list[str],
    lags: list[int],
    windows: list[int],
    step_offset: pd.DateOffset,
) -> dict | None:
    """Hold out the last `horizon` steps and compare ML rollout vs. Holt's method.

    Returns {"ml_mae", "holt_mae", "winner"} or None when there are too few rows
    to run both a seed window and a holdout.
    """
    needed = seed_size + 2 * horizon
    if len(df_work) < needed:
        return None

    holdout_start = len(df_work) - horizon
    y_actual = df_work[meta.target_col].values[holdout_start:].astype(float)

    keep_cols = list(
        {sort_col, meta.target_col} | set(lag_cols) | (set(meta.feature_cols) & set(df_work.columns))
    )
    seed_cur = df_work[keep_cols].iloc[max(0, holdout_start - seed_size):holdout_start].copy()
    ml_preds = _ml_rollout(seed_cur, horizon, pipeline, meta, sort_col, lag_cols, lags, windows, step_offset)

    y_hist = df_work[meta.target_col].values[:holdout_start].astype(float)
    holt_preds = _holt_forecast(y_hist, horizon)

    holt_mae = float(np.mean(np.abs(y_actual - holt_preds)))
    ml_mae = float(np.mean(np.abs(y_actual - np.array(ml_preds)))) if ml_preds else None

    winner = "ml" if (ml_mae is not None and ml_mae <= holt_mae) else "holt"

    return {
        "ml_mae": round(ml_mae, 4) if ml_mae is not None else None,
        "holt_mae": round(holt_mae, 4),
        "winner": winner,
    }


def forecast_with_model(
    df: pd.DataFrame,
    model_id: str,
    horizon: int = 30,
    model_manager: ModelManager | None = None,
) -> dict:
    """Multi-step forecast comparing an ML lag model against Holt's linear baseline.

    Runs a holdout comparison on historical data to pick the more accurate method,
    then returns both forecasts as chart series (ML solid, Holt dashed).
    """
    manager = model_manager or ModelManager()
    try:
        pipeline, meta = manager.load_model(model_id)
    except KeyError:
        return {"error": f"Model '{model_id}' not found in registry."}
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    if meta.task_type != "regression":
        return {"error": "Forecasting requires a regression model. Classification models cannot produce numeric forecasts."}

    lag_config = meta.lag_config
    if not lag_config:
        return {
            "error": (
                "This model was not trained with lag features. Autoregressive forecasting "
                "requires a temporal regression model trained on a datetime column."
            )
        }

    sort_col = lag_config["sort_col"]
    lag_cols = lag_config["lag_cols"]
    lags = lag_config.get("lags", LAG_DEFAULTS)
    windows = lag_config.get("windows", ROLLING_DEFAULTS)

    missing_inputs = [c for c in [sort_col, meta.target_col] + lag_cols if c not in df.columns]
    if missing_inputs:
        return {"error": f"Dataset missing required columns for forecasting: {missing_inputs}"}

    try:
        df_work = df.copy()
        df_work[sort_col] = pd.to_datetime(df_work[sort_col], errors="coerce")
        df_work = df_work.dropna(subset=[sort_col]).sort_values(sort_col).reset_index(drop=True)
    except Exception as exc:
        return {"error": f"Could not parse date column '{sort_col}': {exc}"}

    max_lag = max(max(lags), max(windows) if windows else 0)
    seed_size = max_lag + 5
    if len(df_work) < max_lag:
        return {"error": f"Need at least {max_lag} rows to compute lag features, got {len(df_work)}."}

    step_offset = _infer_step_offset(df_work[sort_col])

    # Holdout comparison: compare ML vs. Holt on held-out historical tail.
    baseline_comparison = _holdout_comparison(
        df_work, horizon, seed_size, pipeline, meta,
        sort_col, lag_cols, lags, windows, step_offset,
    )

    # Full ML autoregressive rollout on the complete history.
    keep_cols = list(
        {sort_col, meta.target_col} | set(lag_cols) | (set(meta.feature_cols) & set(df_work.columns))
    )
    cur = df_work[keep_cols].iloc[-seed_size:].copy()
    halfwidth = meta.conformal_halfwidth
    nd = df_work[sort_col].iloc[-1] + step_offset

    ml_rows: list[dict] = []
    for step in range(1, horizon + 1):
        try:
            feat_df, _ = engineer_lag_features(cur, sort_col, lag_cols, lags=lags, windows=windows)
        except Exception:
            break

        last_feat = feat_df.iloc[[-1]]
        last_raw = cur.iloc[-1]
        X_dict: dict = {}
        for col in meta.feature_cols:
            if col in last_feat.columns:
                X_dict[col] = last_feat[col].values
            elif col in last_raw.index:
                X_dict[col] = [last_raw[col]]
            else:
                X_dict[col] = [np.nan]

        try:
            pred_raw = float(pipeline.predict(pd.DataFrame(X_dict))[0])
        except Exception:
            break

        pred = np.expm1(pred_raw) if getattr(meta, "log_transform_target", False) else pred_raw
        row_out: dict = {"step": step, "date": nd.strftime("%Y-%m-%d"), "prediction": round(float(pred), 4)}
        if halfwidth is not None:
            lower = pred - float(halfwidth)
            if getattr(meta, "log_transform_target", False):
                lower = max(0.0, lower)
            row_out["lower_90"] = round(float(lower), 4)
            row_out["upper_90"] = round(float(pred + float(halfwidth)), 4)
        ml_rows.append(row_out)

        new_row = cur.iloc[[-1]].copy()
        new_row[sort_col] = nd
        new_row[meta.target_col] = pred
        for c in lag_cols:
            if c in new_row.columns and c != meta.target_col:
                new_row[c] = pred
        cur = pd.concat([cur, new_row], ignore_index=True)
        nd = nd + step_offset

    if not ml_rows:
        return {"error": "Could not generate any forecast steps. Verify the model was trained with temporal lag features."}

    # Holt forecast over the full forecast horizon for comparison.
    y_all = df_work[meta.target_col].values.astype(float)
    holt_arr = _holt_forecast(y_all, len(ml_rows))
    holt_rows = [
        {"date": ml_rows[i]["date"], "holt_forecast": round(float(holt_arr[i]), 4)}
        for i in range(len(ml_rows))
    ]

    # Merge ML and Holt into a single chart data list.
    holt_by_date = {r["date"]: r["holt_forecast"] for r in holt_rows}
    chart_data = [
        {**{k: v for k, v in r.items() if k != "step"}, "holt_forecast": holt_by_date.get(r["date"])}
        for r in ml_rows
    ]

    has_pi = "lower_90" in ml_rows[0]
    y_series = ["prediction", "lower_90", "upper_90", "holt_forecast"] if has_pi else ["prediction", "holt_forecast"]
    chart = {
        "type": "line",
        "title": f"{len(ml_rows)}-step forecast: {meta.target_col}",
        "x": "date",
        "y": "prediction",
        "y_series": y_series,
        "data": chart_data,
    }

    winner_note = ""
    if baseline_comparison:
        ml_mae = baseline_comparison.get("ml_mae")
        holt_mae = baseline_comparison["holt_mae"]
        w = "ML" if baseline_comparison["winner"] == "ml" else "Holt"
        winner_note = (
            f" Holdout ({horizon}-step): ML MAE={ml_mae}, Holt MAE={holt_mae}. {w} performed better."
        )

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "horizon": horizon,
        "horizon_steps": len(ml_rows),
        "has_prediction_intervals": has_pi,
        "conformal_halfwidth": halfwidth,
        "forecast_rows": ml_rows,
        "holt_forecast_rows": holt_rows,
        "baseline_comparison": baseline_comparison,
        "charts": [chart],
        "engineering_readout": (
            f"{len(ml_rows)}-step forecast for '{meta.target_col}' using {meta.model_type} "
            f"(solid) vs. Holt's linear baseline (dashed). Step size inferred from data."
            + (f" 90% PIs included (±{halfwidth:.4f})." if has_pi else "")
            + winner_note
        ),
    }
