# app/analytics/quality.py
from __future__ import annotations

import pandas as pd

from app.analytics.viz_specs import histogram_spec, simple_bar_spec

TOP_CATEGORICAL_VALUES = 5
HIGH_MISSING_THRESHOLD = 0.5
HIGH_SKEW_THRESHOLD = 2.0
HIGH_CARDINALITY_RATIO = 0.9
MAX_QUALITY_HISTOGRAMS = 6


def _categorical_top_values(s: pd.Series) -> list[dict]:
    vc = s.value_counts(dropna=True).head(TOP_CATEGORICAL_VALUES)
    total = int(s.notna().sum())
    return [
        {
            "value": str(value),
            "count": int(count),
            "pct": round(float(count) / total * 100, 2) if total else None,
        }
        for value, count in vc.items()
    ]


def data_quality_report(df: pd.DataFrame, sample: int = 10000) -> dict:
    d = df.sample(sample, random_state=42) if len(df) > sample else df
    n_rows = int(df.shape[0])

    columns: list[dict] = []
    findings: list[str] = []
    skewed_numeric_cols: list[tuple[str, float]] = []

    for c in d.columns:
        s = d[c]
        missing_pct = float(s.isna().mean())
        n_unique = int(s.nunique(dropna=True))

        col = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(missing_pct * 100, 2),
            "unique": n_unique,
        }

        if pd.api.types.is_datetime64_any_dtype(s):
            clean = s.dropna()
            if not clean.empty:
                col.update({"min": clean.min().isoformat(), "max": clean.max().isoformat()})
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if not clean.empty:
                col.update(
                    {
                        "mean": float(clean.mean()),
                        "std": float(clean.std()) if len(clean) > 1 else None,
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "p25": float(clean.quantile(0.25)),
                        "p50": float(clean.quantile(0.50)),
                        "p75": float(clean.quantile(0.75)),
                        "p95": float(clean.quantile(0.95)),
                        "p99": float(clean.quantile(0.99)),
                    }
                )
                if len(clean) > 2:
                    skewness = float(clean.skew())
                    col["skewness"] = skewness
                    if abs(skewness) >= HIGH_SKEW_THRESHOLD:
                        skewed_numeric_cols.append((str(c), skewness))
        else:
            col["top_values"] = _categorical_top_values(s)
            # High cardinality is only suspicious for genuinely categorical
            # data; continuous numerics and datetimes are handled above and
            # never reach this branch.
            if len(d) > 0 and n_unique / len(d) >= HIGH_CARDINALITY_RATIO:
                findings.append(f"Column '{c}' has very high cardinality ({n_unique:,} distinct values) — likely free text or an identifier.")

        if missing_pct >= HIGH_MISSING_THRESHOLD:
            findings.append(f"Column '{c}' is {missing_pct * 100:.0f}% missing.")

        columns.append(col)

    for col_name, skewness in skewed_numeric_cols:
        findings.append(f"Column '{col_name}' is highly skewed (skewness={skewness:.2f}).")

    charts = [
        histogram_spec(d, column=col_name, title=f"Distribution of {col_name} (skewness={skewness:.2f})")
        for col_name, skewness in sorted(skewed_numeric_cols, key=lambda item: -abs(item[1]))[:MAX_QUALITY_HISTOGRAMS]
    ]

    if findings:
        readout = f"Quality scan of {n_rows:,} rows × {df.shape[1]} cols found {len(findings)} issue(s): " + " ".join(findings[:5])
        if len(findings) > 5:
            readout += f" (+{len(findings) - 5} more below.)"
    else:
        readout = f"Quality scan of {n_rows:,} rows × {df.shape[1]} cols found no major issues."

    return {
        "n_rows": n_rows,
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "findings": findings,
        "charts": charts,
        "engineering_readout": readout,
    }


def missingness_matrix(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    m = df.isna().mean().sort_values(ascending=False)
    out = m.head(top_n).reset_index()
    out.columns = ["column", "missing_pct"]
    return out


def overrepresented_categories(
        df: pd.DataFrame,
        col: str,
        threshold: float = 0.5,
        top_k: int = 10,
) -> dict:
    if col not in df.columns:
        return {"error": f"Column not found: {col}", "available_columns": list(map(str, df.columns))}

    vc = df[col].value_counts(normalize=True, dropna=False)
    dominant = vc[vc >= threshold].head(top_k)
    top_dist = vc.head(top_k)

    dist_df = pd.DataFrame({"value": [str(v) for v in top_dist.index], "proportion": top_dist.values})
    chart = simple_bar_spec(dist_df, x="value", y="proportion", title=f"Top values in '{col}'")

    if not dominant.empty:
        top_value, top_share = next(iter(dominant.items()))
        readout = f"Column '{col}': '{top_value}' accounts for {top_share * 100:.1f}% of rows."
    else:
        readout = f"Column '{col}': no single value reaches the {threshold * 100:.0f}% dominance threshold."

    return {
        "column": col,
        "dominant_values": dominant.to_dict(),
        "top_distribution": top_dist.to_dict(),
        "charts": [chart],
        "engineering_readout": readout,
    }


def skewed_features(
        df: pd.DataFrame,
        sample: int = 10000,
        threshold: float = 1.0,
        max_features: int = 50,
) -> dict:
    d = df.sample(sample, random_state=42) if len(df) > sample else df

    features: list[dict] = []
    for c in d.columns:
        s = d[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue

        clean = pd.to_numeric(s, errors="coerce").dropna()
        if len(clean) <= 2:
            continue

        sk = float(clean.skew())
        if abs(sk) >= threshold:
            features.append(
                {
                    "column": str(c),
                    "skewness": sk,
                    "severity": "high" if abs(sk) >= HIGH_SKEW_THRESHOLD else "moderate",
                }
            )

    # most skewed first
    features.sort(key=lambda x: abs(x["skewness"]), reverse=True)
    features = features[:max_features]

    charts = [
        histogram_spec(d, column=item["column"], title=f"Distribution of {item['column']} (skewness={item['skewness']:.2f})")
        for item in features[:MAX_QUALITY_HISTOGRAMS]
    ]

    if features:
        readout = f"Found {len(features)} skewed feature(s) (|skewness| ≥ {threshold}); most skewed: {features[0]['column']} ({features[0]['skewness']:.2f})."
    else:
        readout = f"No numeric features with |skewness| ≥ {threshold} were found."

    return {
        "features": features,
        "charts": charts,
        "engineering_readout": readout,
    }
