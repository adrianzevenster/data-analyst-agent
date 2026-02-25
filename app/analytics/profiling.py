from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import skew

def profile_dataset(df: pd.DataFrame, sample: int = 5000) -> dict:
    d = df
    if len(d) > sample:
        d = d.sample(sample, random_state=42)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
    }

    for c in d.columns:
        s = d[c]
        col = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            col.update(
                {
                    "min": float(s.min(skipna=True)) if s.notna().any() else None,
                    "max": float(s.max(skipna=True)) if s.notna().any() else None,
                    "mean": float(s.mean(skipna=True)) if s.notna().any() else None,
                }
            )
        summary["columns"].append(col)

    return summary
