from __future__ import annotations

import pandas as pd
import numpy as np


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Column names
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\t", " ") for c in df.columns]

    # Basic dtype coercions (safe-ish defaults)
    for c in df.columns:
        if df[c].dtype == "object":
            # Try numeric
            s = pd.to_numeric(df[c], errors="ignore")
            df[c] = s

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
