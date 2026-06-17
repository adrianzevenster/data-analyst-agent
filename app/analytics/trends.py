from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app.analytics.viz_specs import line_spec

DATETIME_PARSE_SUCCESS_THRESHOLD = 0.9
LONG_SPAN_DAYS = 400
MEDIUM_SPAN_DAYS = 60


def detect_datetime_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        # Speculatively probing non-date columns (e.g. "high"/"low") is
        # expected to fail often; the success-ratio check below handles
        # that, so suppress pandas' per-column parse-format warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() >= DATETIME_PARSE_SUCCESS_THRESHOLD:
            return str(c)

    return None


def _detect_value_col(df: pd.DataFrame, exclude: str) -> str | None:
    numeric_cols = [
        str(c) for c in df.columns
        if c != exclude and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    return numeric_cols[0] if numeric_cols else None


def _infer_freq(span_days: float) -> str:
    if span_days > LONG_SPAN_DAYS:
        return "M"
    if span_days > MEDIUM_SPAN_DAYS:
        return "W"
    return "D"


def trend_analysis(
    df: pd.DataFrame,
    date_col: str | None = None,
    value_col: str | None = None,
    freq: str = "auto",
    agg: str = "sum",
) -> dict:
    resolved_date_col = date_col or detect_datetime_col(df)
    if not resolved_date_col:
        return {"error": "No datetime column found or specified for trend analysis."}

    resolved_value_col = value_col or _detect_value_col(df, exclude=resolved_date_col)
    if not resolved_value_col:
        return {"error": "No numeric value column found or specified for trend analysis."}

    dates = pd.to_datetime(df[resolved_date_col], errors="coerce")
    values = pd.to_numeric(df[resolved_value_col], errors="coerce")
    d = pd.DataFrame({"date": dates, "value": values}).dropna()

    if d.empty:
        return {"error": "No valid (date, value) rows available for trend analysis."}

    span_days = (d["date"].max() - d["date"].min()).total_seconds() / 86400
    resolved_freq = _infer_freq(span_days) if freq == "auto" else freq

    try:
        series = d.set_index("date")["value"].resample(resolved_freq).agg(agg).dropna()
    except ValueError as exc:
        return {"error": f"Could not resample at freq={resolved_freq!r}: {exc}"}

    if len(series) < 2:
        return {"error": "Not enough distinct time periods to analyze a trend."}

    period_labels = [ts.strftime("%Y-%m-%d") for ts in series.index]
    pct_change = series.pct_change()

    latest_value = float(series.iloc[-1])
    latest_change_pct = float(pct_change.iloc[-1] * 100) if pd.notna(pct_change.iloc[-1]) else None

    peak_idx = series.idxmax()
    trough_idx = series.idxmin()

    # Use the fitted trend line's endpoints (not raw first/last values) for
    # overall change: resample() can produce a partial first/last bucket
    # (e.g. one day in a "week"), which makes raw boundary values noisy and
    # can wildly distort a first-vs-last percentage.
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.to_numpy(dtype=float), 1)
    slope, intercept = float(slope), float(intercept)
    direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

    fitted_first = intercept
    fitted_last = intercept + slope * (len(series) - 1)
    overall_change_pct = ((fitted_last - fitted_first) / abs(fitted_first) * 100) if fitted_first else None

    periods = [
        {
            "period": label,
            "value": float(value),
            "pct_change": float(pc * 100) if pd.notna(pc) else None,
        }
        for label, value, pc in zip(period_labels, series.to_numpy(), pct_change.to_numpy())
    ]

    chart = line_spec(
        pd.DataFrame({"period": period_labels, "value": series.to_numpy()}),
        x="period",
        y="value",
        title=f"{resolved_value_col} over time ({resolved_freq})",
    )

    readout_parts = [
        (
            f"{resolved_value_col} trended {direction} overall ({overall_change_pct:+.1f}% from first to last period)."
            if overall_change_pct is not None
            else f"{resolved_value_col} trend is {direction}."
        ),
        f"Latest period ({period_labels[-1]}): {latest_value:,.2f}"
        + (f", {latest_change_pct:+.1f}% vs previous period." if latest_change_pct is not None else "."),
        f"Peak: {peak_idx.strftime('%Y-%m-%d')} ({float(series.max()):,.2f}). "
        f"Trough: {trough_idx.strftime('%Y-%m-%d')} ({float(series.min()):,.2f}).",
    ]

    return {
        "date_col": resolved_date_col,
        "value_col": resolved_value_col,
        "freq": resolved_freq,
        "agg": agg,
        "n_periods": len(series),
        "periods": periods,
        "direction": direction,
        "overall_change_pct": round(overall_change_pct, 2) if overall_change_pct is not None else None,
        "latest_value": latest_value,
        "latest_change_pct": round(latest_change_pct, 2) if latest_change_pct is not None else None,
        "peak_period": peak_idx.strftime("%Y-%m-%d"),
        "peak_value": float(series.max()),
        "trough_period": trough_idx.strftime("%Y-%m-%d"),
        "trough_value": float(series.min()),
        "charts": [chart],
        "engineering_readout": " ".join(readout_parts),
    }
