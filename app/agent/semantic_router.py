"""Semantic routing layer: embeds tool descriptions and matches queries by cosine similarity.

Used as a fallback when the rule-based planner produces no match, before resorting to
the generic auto_insights catch-all. Embeddings are computed once at first use via the
shared LocalEmbedder singleton (all-MiniLM-L6-v2).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTIONS: list[tuple[str, str]] = [
    ("profile_dataset",            "Summarize dataset columns statistics distributions schema overview data types"),
    ("data_quality_report",        "Check data quality find duplicates missing values null analysis data health completeness"),
    ("auto_insights",              "Find key insights interesting patterns surprising findings notable trends"),
    ("correlation_analysis",       "Compute correlations relationships between variables association strength"),
    ("trend_analysis",             "Analyze trends over time time series seasonality growth rates month over month"),
    ("kmeans_clusters",            "Cluster data find segments group similar rows natural groupings customer segments"),
    ("anomaly_scan",               "Detect anomalies outliers unusual rows flag suspicious data points"),
    ("train_supervised_model",     "Train machine learning model build predictor fit classifier regressor supervised learning"),
    ("score_with_model",           "Apply trained model score data generate predictions inference"),
    ("explain_model",              "Explain model feature importance what features matter model explainability permutation importance"),
    ("shap_explain_prediction",    "SHAP local explanation why did model predict this specific row contribution"),
    ("duckdb_query",               "Run SQL query aggregate filter custom SQL select group by"),
    ("forecast_with_model",        "Forecast future values predict next steps time series forecasting"),
    ("hypothesis_test",            "Statistical hypothesis test t-test chi-squared ANOVA significance test"),
    ("estimate_causal_effect",     "Causal inference effect of treatment impact analysis causal effect"),
    ("multidim_pivot",             "Pivot table group by multiple dimensions breakdown category aggregation"),
    ("cross_dataset_profile",      "Compare datasets cross-dataset analysis join tables multi-dataset relationships"),
    ("missingness_matrix",         "Missing values null heatmap incomplete data NaN analysis"),
    ("overrepresented_categories", "Class imbalance dominant category bias overrepresented values"),
]

# Minimum cosine similarity to accept a semantic match.
_THRESHOLD = 0.40

_embeddings: np.ndarray | None = None
_tool_names: list[str] = []


def _get_embeddings() -> tuple[np.ndarray, list[str]]:
    global _embeddings, _tool_names
    if _embeddings is not None:
        return _embeddings, _tool_names
    try:
        from app.rag.embedder import LocalEmbedder
        emb = LocalEmbedder()
        descs = [desc for _, desc in _TOOL_DESCRIPTIONS]
        # LocalEmbedder already returns normalized vectors.
        vecs = emb.embed(descs)
        _embeddings = np.array(vecs, dtype=np.float32)
        _tool_names = [name for name, _ in _TOOL_DESCRIPTIONS]
        logger.info("SemanticRouter: embedded %d tool descriptions", len(_tool_names))
    except Exception as exc:
        logger.warning("SemanticRouter: embedding failed (%s), router disabled", exc)
        _embeddings = np.empty((0, 1), dtype=np.float32)
        _tool_names = []
    return _embeddings, _tool_names


def route(query: str, top_k: int = 1) -> list[str]:
    """Return up to top_k tool names whose description best matches query (above threshold).

    Returns an empty list when the embedder is unavailable or no tool clears the threshold.
    """
    emb_matrix, names = _get_embeddings()
    if emb_matrix.shape[0] == 0:
        return []
    try:
        from app.rag.embedder import LocalEmbedder
        q_vec = np.array(LocalEmbedder().embed([query])[0], dtype=np.float32)
        # Vectors from LocalEmbedder are already normalized; guard against edge cases.
        norm = float(np.linalg.norm(q_vec))
        if norm > 0:
            q_vec = q_vec / norm
        scores = emb_matrix @ q_vec
        best_idxs = np.argsort(scores)[::-1][:top_k]
        return [names[i] for i in best_idxs if float(scores[i]) >= _THRESHOLD]
    except Exception as exc:
        logger.debug("SemanticRouter.route failed: %s", exc)
        return []
