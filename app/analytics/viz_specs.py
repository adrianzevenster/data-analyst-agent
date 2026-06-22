from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_records(records: list[dict]) -> list[dict]:
    """Convert numpy scalars to native Python types so records are JSON-safe."""
    coerced = []
    for row in records:
        safe = {}
        for k, v in row.items():
            if isinstance(v, np.integer):
                safe[k] = int(v)
            elif isinstance(v, np.floating):
                safe[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, np.bool_):
                safe[k] = bool(v)
            elif isinstance(v, float) and np.isnan(v):
                safe[k] = None
            else:
                safe[k] = v
        coerced.append(safe)
    return coerced


def simple_bar_spec(df: pd.DataFrame, x: str, y: str, title: str = "") -> dict:
    data = _coerce_records(df[[x, y]].head(200).to_dict(orient="records"))
    return {
        "type": "bar",
        "title": title or f"{y} by {x}",
        "x": x,
        "y": y,
        "x_label": x,
        "y_label": y,
        "data": data,
    }


def multi_series_bar_spec(df: pd.DataFrame, x: str, y_cols: list[str], title: str = "") -> dict:
    """Bar chart with multiple value series sharing the same x-axis category.

    Useful for pivot/groupby outputs with more than one aggregated value
    column, where picking just the first numeric column would silently
    discard the rest.
    """
    cols = [x] + [c for c in y_cols if c in df.columns]
    data = _coerce_records(df[cols].head(200).to_dict(orient="records"))
    return {
        "type": "bar",
        "title": title or f"{', '.join(y_cols)} by {x}",
        "x": x,
        "y": y_cols[0] if len(y_cols) == 1 else None,
        "y_series": y_cols,
        "x_label": x,
        "y_label": y_cols[0] if len(y_cols) == 1 else "value",
        "data": data,
    }


def histogram_spec(df: pd.DataFrame, column: str, bins: int = 20, title: str = "") -> dict:
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return {
            "type": "histogram",
            "title": title or f"Distribution of {column}",
            "column": column,
            "x_label": column,
            "y_label": "count",
            "data": [],
        }

    counts, edges = np.histogram(series, bins=bins)
    data = [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "bin_label": f"{edges[i]:.2g}–{edges[i + 1]:.2g}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return {
        "type": "histogram",
        "title": title or f"Distribution of {column}",
        "column": column,
        "x_label": column,
        "y_label": "count",
        "data": data,
    }


def line_spec(df: pd.DataFrame, x: str, y: str, title: str = "") -> dict:
    d = df[[x, y]].dropna().sort_values(x).head(2000)
    data = _coerce_records(d.assign(**{x: d[x].astype(str)}).to_dict(orient="records"))
    return {
        "type": "line",
        "title": title or f"{y} over {x}",
        "x": x,
        "y": y,
        "x_label": x,
        "y_label": y,
        "data": data,
    }


def scatter_spec(df: pd.DataFrame, x: str, y: str, title: str = "") -> dict:
    d = df[[x, y]].dropna()

    correlation = None
    if pd.api.types.is_numeric_dtype(d[x]) and pd.api.types.is_numeric_dtype(d[y]) and len(d) > 1:
        correlation = float(d[x].corr(d[y]))

    data = _coerce_records(d.head(2000).to_dict(orient="records"))
    return {
        "type": "scatter",
        "title": title or f"{y} vs {x}",
        "x": x,
        "y": y,
        "x_label": x,
        "y_label": y,
        "correlation": correlation,
        "data": data,
    }
