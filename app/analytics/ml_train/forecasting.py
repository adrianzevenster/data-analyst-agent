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


def forecast_with_model(
    df: pd.DataFrame,
    model_id: str,
    horizon: int = 30,
    model_manager: ModelManager | None = None,
) -> dict:
    """Multi-step autoregressive forecast using a lag-feature regression model.

    Requires a model trained with lag features (temporal split). Rolls the model
    forward `horizon` steps by re-engineering lag features after each prediction
    and feeding the predicted target value back into the lag window.

    Returns forecast rows, a line chart spec with 90% prediction intervals when
    available, and an engineering_readout summary.
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

    # Parse and sort by date
    try:
        df_work = df.copy()
        df_work[sort_col] = pd.to_datetime(df_work[sort_col], errors="coerce")
        df_work = df_work.dropna(subset=[sort_col]).sort_values(sort_col).reset_index(drop=True)
    except Exception as exc:
        return {"error": f"Could not parse date column '{sort_col}': {exc}"}

    max_lag = max(max(lags), max(windows) if windows else 0)
    seed_size = max_lag + 5  # a few extra rows so the first lag window is stable
    if len(df_work) < max_lag:
        return {"error": f"Need at least {max_lag} rows to compute lag features, got {len(df_work)}."}

    step_offset = _infer_step_offset(df_work[sort_col])

    # Seed window: copy only the columns we'll actually need
    keep_cols = list({sort_col, meta.target_col} | set(lag_cols) | set(meta.feature_cols) & set(df_work.columns))
    current = df_work[keep_cols].iloc[-seed_size:].copy()

    last_date = df_work[sort_col].iloc[-1]
    next_date = last_date + step_offset
    halfwidth = meta.conformal_halfwidth

    forecast_rows: list[dict] = []

    for step in range(1, horizon + 1):
        # Re-engineer lag features from the current window
        try:
            feat_df, _ = engineer_lag_features(
                current, sort_col, lag_cols, lags=lags, windows=windows
            )
        except Exception:
            break

        # Build the feature vector for the last row (the most recent timestep)
        last_feat = feat_df.iloc[[-1]]

        # Align to the feature set the model expects, filling missing cols from
        # the last observed row in current (e.g. non-lag numeric/categorical cols).
        X_dict = {}
        last_raw = current.iloc[-1]
        for col in meta.feature_cols:
            if col in last_feat.columns:
                X_dict[col] = last_feat[col].values
            elif col in last_raw.index:
                X_dict[col] = [last_raw[col]]
            else:
                X_dict[col] = [np.nan]
        X_step = pd.DataFrame(X_dict)

        try:
            pred_raw = float(pipeline.predict(X_step)[0])
        except Exception:
            break

        pred = np.expm1(pred_raw) if meta.log_transform_target else pred_raw

        row_out: dict = {
            "step": step,
            "date": next_date.strftime("%Y-%m-%d"),
            "prediction": round(float(pred), 4),
        }
        if halfwidth is not None:
            lower = pred - float(halfwidth)
            if meta.log_transform_target:
                lower = max(0.0, lower)
            row_out["lower_90"] = round(float(lower), 4)
            row_out["upper_90"] = round(float(pred + float(halfwidth)), 4)

        forecast_rows.append(row_out)

        # Append the predicted row back into the working window so the next
        # step's lag features see this prediction as a recent observation.
        new_row = current.iloc[[-1]].copy()
        new_row = new_row.copy()
        new_row[sort_col] = next_date
        new_row[meta.target_col] = pred
        # Update other lag source cols to the predicted value (best-effort;
        # exogenous cols hold their last observed value automatically via copy).
        for c in lag_cols:
            if c in new_row.columns and c != meta.target_col:
                new_row[c] = pred
        current = pd.concat([current, new_row], ignore_index=True)
        next_date = next_date + step_offset

    if not forecast_rows:
        return {"error": "Could not generate any forecast steps. Verify the model was trained with temporal lag features."}

    has_pi = "lower_90" in forecast_rows[0]
    y_series = ["prediction", "lower_90", "upper_90"] if has_pi else ["prediction"]
    chart = {
        "type": "line",
        "title": f"{len(forecast_rows)}-step forecast: {meta.target_col}",
        "x": "date",
        "y": "prediction",
        "y_series": y_series,
        "data": [{k: v for k, v in r.items() if k != "step"} for r in forecast_rows],
    }

    return {
        "model_id": model_id,
        "task_type": meta.task_type,
        "model_type": meta.model_type,
        "target_col": meta.target_col,
        "horizon": horizon,
        "horizon_steps": len(forecast_rows),
        "has_prediction_intervals": has_pi,
        "conformal_halfwidth": halfwidth,
        "forecast_rows": forecast_rows,
        "charts": [chart],
        "engineering_readout": (
            f"{len(forecast_rows)}-step autoregressive forecast for '{meta.target_col}' "
            f"using {meta.model_type}. Step size inferred from data. "
            + (f"90% prediction intervals included (±{halfwidth:.4f})." if has_pi else "No prediction intervals (model was not evaluated on a test set large enough for conformal calibration).")
        ),
    }
