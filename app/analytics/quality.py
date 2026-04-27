# app/analytics/quality.py
from __future__ import annotations

import pandas as pd


def data_quality_report(df: pd.DataFrame, sample: int = 10000) -> dict:
    d = df.sample(sample, random_state=42) if len(df) > sample else df

    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
    }

    for c in d.columns:
        s = d[c]
        col = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "unique": int(s.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(s):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            col.update(
                {
                    "mean": float(clean.mean()) if not clean.empty else None,
                    "std": float(clean.std()) if not clean.empty else None,
                    "min": float(clean.min()) if not clean.empty else None,
                    "max": float(clean.max()) if not clean.empty else None,
                    "skewness": float(clean.skew()) if len(clean) > 2 else None,
                    "p95": float(clean.quantile(0.95)) if not clean.empty else None,
                    "p99": float(clean.quantile(0.99)) if not clean.empty else None,
                }
            )

        report["columns"].append(col)

    return report


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

    return {
        "column": col,
        "dominant_values": vc[vc >= threshold].head(top_k).to_dict(),
        "top_distribution": vc.head(top_k).to_dict(),
    }


def skewed_features(
        df: pd.DataFrame,
        sample: int = 10000,
        threshold: float = 1.0,
        max_features: int = 50,
) -> list[dict]:
    d = df.sample(sample, random_state=42) if len(df) > sample else df

    out: list[dict] = []
    for c in d.columns:
        s = d[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        clean = pd.to_numeric(s, errors="coerce").dropna()
        if len(clean) <= 2:
            continue

        sk = float(clean.skew())
        if abs(sk) >= threshold:
            out.append(
                {
                    "column": str(c),
                    "skewness": sk,
                    "severity": "high" if abs(sk) >= 2 else "moderate",
                }
            )

    # most skewed first
    out.sort(key=lambda x: abs(x["skewness"]), reverse=True)
    return out[:max_features]
