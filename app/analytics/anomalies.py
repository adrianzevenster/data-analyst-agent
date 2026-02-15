from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def anomaly_scan(df: pd.DataFrame, numeric_cols: list[str], contamination: float = 0.02, sample: int = 20000) -> pd.DataFrame:
    d = df[numeric_cols].copy()
    d = d.dropna()
    if len(d) == 0:
        return pd.DataFrame({"error": ["No non-null rows for selected numeric columns."]})

    if len(d) > sample:
        d = d.sample(sample, random_state=42)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    scores = model.fit_predict(d)
    # -1 anomaly, 1 normal
    out = d.copy()
    out["is_anomaly"] = (scores == -1)
    return out[out["is_anomaly"]].head(200)
