from __future__ import annotations

import os
import re
from app.core.models import ToolCall


class Planner:
    """
    Lightweight rule + retrieval assisted planner.
    Retrieval is lazy-initialized so missing optional deps don't crash the API.
    """

    def __init__(self):
        self._rag = None  # lazy

    def _get_rag(self):
        if self._rag is None:
            from app.rag.retriever import RagRetriever
            self._rag = RagRetriever()
        return self._rag

    def plan(self, message: str, dataset_id: str | None, top_k: int = 6) -> tuple[list[ToolCall], list[dict]]:
        m = (message or "").lower()
        calls: list[ToolCall] = []

        # Allow turning retrieval off (useful for dev / CI)
        enable_rag = os.getenv("ENABLE_RAG", "1") == "1"
        citations = self._get_rag().retrieve(message, top_k=top_k) if enable_rag else []

        # Profile / schema
        if any(k in m for k in ["profile", "summary", "overview", "columns", "schema"]):
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        # Data quality / diagnostics
        if any(k in m for k in ["quality", "data quality", "diagnostic", "diagnostics", "healthcheck", "health check"]):
            calls.append(ToolCall(name="data_quality_report", arguments={"sample": 10000}))

        # Missing values
        if any(k in m for k in ["missing", "null", "nan", "incomplete"]):
            calls.append(ToolCall(name="missingness_matrix", arguments={"top_n": 30}))

        # Skewness / heavy tails
        if any(k in m for k in ["skew", "skewed", "long tail", "heavy tail"]):
            calls.append(ToolCall(name="skewed_features", arguments={"sample": 10000, "threshold": 1.0}))

        # Over-representation / imbalance
        if any(k in m for k in ["overrepresented", "over-represented", "dominant", "imbalance", "class imbalance", "bias"]):
            col_match = re.search(r"(?:in|for|column)\s+([a-zA-Z0-9_]+)", message, flags=re.IGNORECASE)
            if col_match:
                calls.append(
                    ToolCall(
                        name="overrepresented_categories",
                        arguments={"col": col_match.group(1), "threshold": 0.5, "top_k": 10},
                    )
                )
            else:
                calls.append(ToolCall(name="data_quality_report", arguments={"sample": 10000}))

        # Pivot / cube-like
        if "pivot" in m or "group by" in m or "breakdown" in m:
            dims = re.findall(r"by\s+([a-zA-Z0-9_ ,]+)", message, flags=re.IGNORECASE)
            index: list[str] = []
            if dims:
                index = [d.strip() for d in re.split(r"[,\sand]+", dims[0]) if d.strip()]
            calls.append(
                ToolCall(
                    name="multidim_pivot",
                    arguments={"index": index[:3] or [], "columns": None, "values": "", "agg": "sum", "top_n": 50},
                )
            )

        # SQL tool
        if "sql:" in m:
            q = message.split("sql:", 1)[1].strip()
            calls.append(ToolCall(name="duckdb_query", arguments={"query": q}))

        # Anomalies / outliers
        if "anomal" in m or "outlier" in m:
            calls.append(ToolCall(name="anomaly_scan", arguments={"numeric_cols": [], "contamination": 0.02}))

        # Clustering / segmentation
        if "cluster" in m or "segmentation" in m:
            calls.append(ToolCall(name="kmeans_clusters", arguments={"numeric_cols": [], "k": 5}))

        if not calls and dataset_id:
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        return calls, citations
