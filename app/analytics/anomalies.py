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


def explain_anomaly(
    df: pd.DataFrame,
    numeric_cols: list[str],
    row_idx: int = 0,
    top_k: int = 10,
) -> dict:
    """Explain why a specific row is anomalous using percentile attribution.

    For each numeric feature, computes the row's percentile in the full dataset
    and how extreme it is (distance from the 50th percentile). Returns the top-K
    most extreme features sorted by extremeness.
    """
    if row_idx < 0 or row_idx >= len(df):
        return {"error": f"row_idx {row_idx} is out of range for dataset with {len(df)} rows."}

    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return {"error": "No valid numeric columns found."}

    row = df.iloc[row_idx]
    attributions = []

    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        val = pd.to_numeric(row[col], errors="coerce")
        if len(series) < 2 or pd.isna(val):
            continue

        percentile = float((series < val).mean() * 100)
        # How far from median — 50th pct → 0, 0th or 100th → 50
        extremeness = abs(percentile - 50.0)
        mean = float(series.mean())
        std = float(series.std())
        z_score = (float(val) - mean) / max(std, 1e-9)

        direction = "high" if percentile > 50 else "low"
        attributions.append({
            "feature": col,
            "value": round(float(val), 6),
            "percentile": round(percentile, 1),
            "extremeness_pct": round(extremeness, 1),
            "z_score": round(z_score, 2),
            "direction": direction,
            "population_mean": round(mean, 4),
            "population_std": round(std, 4),
        })

    attributions.sort(key=lambda x: -float(str(x["extremeness_pct"] or 0)))
    top = attributions[:top_k]

    if not top:
        return {"error": "No numeric features could be attributed for this row."}

    readout_parts = []
    for a in top[:3]:
        readout_parts.append(
            f"'{a['feature']}' is {a['direction']} (value: {a['value']}, "
            f"{a['percentile']:.0f}th percentile, z={a['z_score']:.1f})"
        )

    return {
        "row_idx": row_idx,
        "n_features_checked": len(cols),
        "top_attributions": top,
        "engineering_readout": (
            f"Row {row_idx} anomaly explained: {len(top)} features checked. "
            f"Most extreme: {'; '.join(readout_parts)}."
        ),
    }
