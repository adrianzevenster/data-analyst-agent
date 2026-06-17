from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def anomaly_scan(df: pd.DataFrame, numeric_cols: list[str], contamination: float = 0.02, sample: int = 20000) -> dict:
    d = df[numeric_cols].copy()
    d = d.dropna()
    if len(d) == 0:
        return {"error": "No non-null rows for selected numeric columns."}

    n_rows_available = len(d)
    if len(d) > sample:
        d = d.sample(sample, random_state=42)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(d)
    # decision_function: lower score = more anomalous. predict: -1 anomaly, 1 normal.
    scores = model.decision_function(d)
    preds = model.predict(d)

    out = d.copy()
    out["anomaly_score"] = scores
    out["is_anomaly"] = preds == -1

    n_anomalies = int(out["is_anomaly"].sum())
    top_anomalies = (
        out[out["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(200)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    return {
        "n_rows_scanned": len(out),
        "n_rows_available": n_rows_available,
        "n_anomalies": n_anomalies,
        "anomaly_rate_pct": round(n_anomalies / len(out) * 100, 2),
        "contamination_param": contamination,
        "top_anomalies": top_anomalies,
        "engineering_readout": (
            f"IsolationForest anomaly scan complete: flagged {n_anomalies} of {len(out)} rows "
            f"({n_anomalies / len(out) * 100:.2f}%) as anomalous across {len(numeric_cols)} feature(s), "
            f"ranked by severity (most anomalous first)."
        ),
    }
