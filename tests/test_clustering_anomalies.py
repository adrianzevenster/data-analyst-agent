from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics.anomalies import anomaly_scan
from app.analytics.clustering import kmeans_clusters


def _two_blob_df(n_per_blob: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=0.0, scale=0.5, size=(n_per_blob, 2))
    blob_b = rng.normal(loc=20.0, scale=0.5, size=(n_per_blob, 2))
    data = np.vstack([blob_a, blob_b])
    return pd.DataFrame(data, columns=["x", "y"])


def test_kmeans_clusters_returns_summary_and_quality_metric():
    df = _two_blob_df()

    result = kmeans_clusters(df, numeric_cols=["x", "y"], k=2)

    assert result["k_used"] == 2
    assert result["n_rows_clustered"] == len(df)
    assert len(result["cluster_summary"]) == 2
    assert sum(c["size"] for c in result["cluster_summary"]) == len(df)
    # Two well-separated blobs should score highly on silhouette.
    assert result["silhouette_score"] > 0.8
    assert "well-separated" in result["engineering_readout"]


def test_kmeans_clusters_clamps_k_above_available_rows():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    result = kmeans_clusters(df, numeric_cols=["x"], k=50)

    assert result["k_requested"] == 50
    assert result["k_used"] == 2
    assert result["n_rows_clustered"] == 3


def test_kmeans_clusters_handles_empty_input():
    df = pd.DataFrame({"x": [None, None]})

    result = kmeans_clusters(df, numeric_cols=["x"], k=5)

    assert "error" in result


def test_anomaly_scan_flags_outliers_and_ranks_by_severity():
    rng = np.random.default_rng(7)
    normal = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    outliers = np.array([[50.0, 50.0], [-50.0, -50.0]])
    df = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b"])

    result = anomaly_scan(df, numeric_cols=["a", "b"], contamination=0.03)

    assert result["n_anomalies"] >= 1
    assert result["n_rows_scanned"] == len(df)
    assert result["anomaly_rate_pct"] > 0
    top = result["top_anomalies"]
    assert len(top) == result["n_anomalies"]
    # Most anomalous (lowest decision_function score) should be sorted first.
    scores = [row["anomaly_score"] for row in top]
    assert scores == sorted(scores)


def test_anomaly_scan_handles_empty_input():
    df = pd.DataFrame({"a": [None, None]})

    result = anomaly_scan(df, numeric_cols=["a"])

    assert "error" in result
