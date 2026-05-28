from __future__ import annotations

import os
import re

import pandas as pd

from app.analytics.dataset_manager import DatasetManager
from app.core.models import ToolCall
from app.agent.llm import LLMReasoner, LLMUnavailable


class Planner:
    """
    Lightweight rule + retrieval assisted planner.
    Retrieval is lazy-initialized so missing optional deps don't crash the API.
    """

    def __init__(self):
        self._rag = None
        self.dm = DatasetManager()
        self.llm = LLMReasoner()

    def _get_rag(self):
        if self._rag is None:
            from app.rag.retriever import RagRetriever
            self._rag = RagRetriever()
        return self._rag

    def _detect_ml_dataset(self, df: pd.DataFrame) -> str | None:
        cols = list(df.columns)

        has_probability_col = any("prob" in str(c).lower() or "score" in str(c).lower() for c in cols)
        has_prediction_col = any(
            any(k in str(c).lower() for k in ["prediction", "predicted", "churn", "classification"])
            for c in cols
        )
        has_actual_col = any(
            any(k in str(c).lower() for k in ["actual", "target", "label", "ground_truth", "ground truth"])
            for c in cols
        )

        if has_probability_col and has_prediction_col and not has_actual_col:
            return "scored_predictions"

        if has_actual_col and has_prediction_col:
            actual_cols = [
                c for c in cols
                if any(k in str(c).lower() for k in ["actual", "target", "label", "ground_truth", "ground truth"])
            ]
            pred_cols = [
                c for c in cols
                if any(k in str(c).lower() for k in ["prediction", "predicted", "classification"])
            ]

            if actual_cols and pred_cols:
                actual_numeric = pd.to_numeric(df[actual_cols[0]], errors="coerce").notna().mean() >= 0.90
                pred_numeric = pd.to_numeric(df[pred_cols[0]], errors="coerce").notna().mean() >= 0.90

                if actual_numeric and pred_numeric:
                    return "regression"

            return "classification"

        metric_like_cols = [
            c for c in cols
            if any(k in str(c).lower() for k in ["wmape", "wbias", "mae", "rmse", "mape"])
        ]
        if metric_like_cols:
            return "auto"

        return None

    def _load_dataset_sample(self, dataset_id: str | None) -> pd.DataFrame | None:
        if not dataset_id:
            return None

        try:
            return self.dm.load_df(dataset_id, limit=5000)
        except Exception:
            return None

    def _rule_plan(self, message: str, dataset_id: str | None, df: pd.DataFrame | None) -> list[ToolCall]:
        m = (message or "").lower()
        calls: list[ToolCall] = []

        ml_requested = any(k in m for k in [
            "evaluate_ml_predictions",
            "ml evaluation",
            "model evaluation",
            "evaluate model",
            "model performance",
            "classification metrics",
            "confusion matrix",
            "roc auc",
            "f1",
            "precision",
            "recall",
            "prediction quality",
            "churn probability",
            "predictions",
        ])

        if ml_requested and df is not None:
            task_hint = self._detect_ml_dataset(df) or "auto"
            calls.append(
                ToolCall(
                    name="evaluate_ml_predictions",
                    arguments={"task_hint": task_hint},
                )
            )

        if any(k in m for k in ["profile", "summary", "overview", "columns", "schema"]):
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        if any(k in m for k in ["quality", "data quality", "diagnostic", "diagnostics", "healthcheck", "health check"]):
            calls.append(ToolCall(name="data_quality_report", arguments={"sample": 10000}))

        if any(k in m for k in ["missing", "null", "nan", "incomplete"]):
            calls.append(ToolCall(name="missingness_matrix", arguments={"top_n": 30}))

        if any(k in m for k in ["skew", "skewed", "long tail", "heavy tail"]):
            calls.append(ToolCall(name="skewed_features", arguments={"sample": 10000, "threshold": 1.0}))

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

        if "sql:" in m:
            q = message.split("sql:", 1)[1].strip()
            calls.append(ToolCall(name="duckdb_query", arguments={"query": q}))

        if "anomal" in m or "outlier" in m:
            calls.append(ToolCall(name="anomaly_scan", arguments={"numeric_cols": [], "contamination": 0.02}))

        if "cluster" in m or "segmentation" in m:
            calls.append(ToolCall(name="kmeans_clusters", arguments={"numeric_cols": [], "k": 5}))

        if not calls and df is not None:
            task_hint = self._detect_ml_dataset(df)
            if task_hint:
                calls.append(
                    ToolCall(
                        name="evaluate_ml_predictions",
                        arguments={"task_hint": task_hint},
                    )
                )

        if not calls and dataset_id:
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        return calls

    def plan(self, message: str, dataset_id: str | None, top_k: int = 6) -> tuple[list[ToolCall], list[dict]]:
        df = self._load_dataset_sample(dataset_id)

        enable_rag = os.getenv("ENABLE_RAG", "1") == "1"
        citations = self._get_rag().retrieve(message, top_k=top_k) if enable_rag else []

        if self.llm.enabled:
            try:
                calls = self.llm.plan(message, dataset_id=dataset_id, df=df, citations=citations)
                if calls:
                    return calls, citations
            except LLMUnavailable:
                pass

        calls = self._rule_plan(message, dataset_id, df)
        return calls, citations
