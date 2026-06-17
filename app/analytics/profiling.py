from __future__ import annotations

import pandas as pd

from app.analytics.viz_specs import histogram_spec

MAX_PROFILE_HISTOGRAMS = 6
TOP_CATEGORICAL_VALUES = 5
HIGH_MISSING_THRESHOLD = 0.5
HIGH_SKEW_THRESHOLD = 2.0


def _numeric_summary(s: pd.Series) -> dict:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {}

    summary = {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else None,
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
    }
    if len(clean) > 2:
        summary["skewness"] = float(clean.skew())
    return summary


def _categorical_summary(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=True).head(TOP_CATEGORICAL_VALUES)
    total = int(s.notna().sum())
    return {
        "top_values": [
            {
                "value": str(value),
                "count": int(count),
                "pct": round(float(count) / total * 100, 2) if total else None,
            }
            for value, count in vc.items()
        ]
    }


def _datetime_summary(s: pd.Series) -> dict:
    clean = s.dropna()
    if clean.empty:
        return {}
    return {
        "min": clean.min().isoformat(),
        "max": clean.max().isoformat(),
        "range_days": float((clean.max() - clean.min()).total_seconds() / 86400),
    }


def _boolean_summary(s: pd.Series) -> dict:
    clean = s.dropna()
    return {
        "true_count": int(clean.sum()),
        "false_count": int((~clean).sum()),
    }


def profile_dataset(df: pd.DataFrame, sample: int = 5000) -> dict:
    d = df
    if len(d) > sample:
        d = d.sample(sample, random_state=42)

    n_rows = int(df.shape[0])
    columns: list[dict] = []
    findings: list[str] = []
    numeric_cols_for_charts: list[str] = []

    for c in d.columns:
        s = d[c]
        n_unique = int(s.nunique(dropna=True))
        missing_pct = float(s.isna().mean())

        col = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(missing_pct * 100, 2),
            "unique": n_unique,
        }

        if pd.api.types.is_bool_dtype(s):
            col.update(_boolean_summary(s))
        elif pd.api.types.is_datetime64_any_dtype(s):
            col.update(_datetime_summary(s))
        elif pd.api.types.is_numeric_dtype(s):
            col.update(_numeric_summary(s))
            numeric_cols_for_charts.append(str(c))
        else:
            col.update(_categorical_summary(s))

        if missing_pct >= HIGH_MISSING_THRESHOLD:
            findings.append(f"Column '{c}' is {missing_pct * 100:.0f}% missing.")

        # Continuous floats are expected to be all-unique; that's not
        # identifier-like, so only flag integer/categorical columns here.
        looks_like_id = n_unique == len(d) and len(d) > 1 and not pd.api.types.is_float_dtype(s)

        if looks_like_id:
            findings.append(f"Column '{c}' looks like an identifier (all values unique).")
        elif n_unique <= 1 and len(d) > 1:
            findings.append(f"Column '{c}' is constant (only one distinct value).")

        skewness = col.get("skewness")
        if skewness is not None and isinstance(skewness, (int, float)) and abs(skewness) >= HIGH_SKEW_THRESHOLD:
            findings.append(f"Column '{c}' is highly skewed (skewness={skewness:.2f}).")

        columns.append(col)

    charts = [
        histogram_spec(d, column=col, title=f"Distribution of {col}")
        for col in numeric_cols_for_charts[:MAX_PROFILE_HISTOGRAMS]
    ]

    if findings:
        readout = f"Profiled {n_rows:,} rows × {df.shape[1]} cols. " + " ".join(findings[:5])
        if len(findings) > 5:
            readout += f" (+{len(findings) - 5} more finding(s) below.)"
    else:
        readout = f"Profiled {n_rows:,} rows × {df.shape[1]} cols. No major structural issues detected."

    return {
        "n_rows": n_rows,
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "findings": findings,
        "charts": charts,
        "engineering_readout": readout,
    }
