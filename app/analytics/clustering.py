from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def kmeans_clusters(df: pd.DataFrame, numeric_cols: list[str], k: int = 5, sample: int = 30000) -> pd.DataFrame:
    d = df[numeric_cols].copy().dropna()
    if len(d) == 0:
        return pd.DataFrame({"error": ["No non-null rows for selected numeric columns."]})
    if len(d) > sample:
        d = d.sample(sample, random_state=42)

    X = StandardScaler().fit_transform(d.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    out = d.copy()
    out["cluster"] = labels
    return out
