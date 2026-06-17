from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_clusters(df: pd.DataFrame, numeric_cols: list[str], k: int = 5, sample: int = 30000) -> dict:
    d = df[numeric_cols].copy().dropna()
    if len(d) == 0:
        return {"error": "No non-null rows for selected numeric columns."}

    n_rows_available = len(d)
    if n_rows_available > sample:
        d = d.sample(sample, random_state=42)

    # KMeans needs at least 2 clusters and strictly fewer clusters than points.
    effective_k = max(2, min(k, len(d) - 1))

    X = StandardScaler().fit_transform(d.values)
    km = KMeans(n_clusters=effective_k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    sil_score = float(silhouette_score(X, labels)) if 2 <= effective_k < len(d) else None

    out = d.copy()
    out["cluster"] = labels

    sizes = out["cluster"].value_counts().sort_index()
    centroids = out.groupby("cluster")[numeric_cols].mean()

    cluster_summary = [
        {
            "cluster": int(cluster_id),
            "size": int(sizes[cluster_id]),
            "pct_of_clustered_rows": round(float(sizes[cluster_id]) / len(out) * 100, 2),
            **{f"mean_{col}": float(centroids.loc[cluster_id, col]) for col in numeric_cols},
        }
        for cluster_id in sizes.index
    ]

    if sil_score is None:
        separation = "n/a"
    elif sil_score > 0.5:
        separation = "well-separated"
    elif sil_score > 0.25:
        separation = "moderately separated"
    else:
        separation = "weak/overlapping structure"

    return {
        "k_requested": k,
        "k_used": effective_k,
        "n_rows_clustered": len(out),
        "n_rows_available": n_rows_available,
        "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
        "cluster_summary": cluster_summary,
        "assignments": out.reset_index(drop=True).head(500).to_dict(orient="records"),
        "engineering_readout": (
            f"KMeans clustering complete: k={effective_k} clusters over {len(numeric_cols)} numeric "
            f"feature(s) across {len(out)} rows. Silhouette score "
            f"{f'{sil_score:.3f}' if sil_score is not None else 'n/a'} ({separation})."
        ),
    }
