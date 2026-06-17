from __future__ import annotations

import os
import re

import pandas as pd

from app.analytics.dataset_manager import DatasetManager
from app.core.models import ToolCall
from app.agent.llm import LLMReasoner, LLMUnavailable

# Phrase -> model_type, checked in order (most specific phrase first so e.g.
# "gradient boosted" matches before a hypothetical shorter overlapping
# phrase would). Values are the generic family aliases from training.py
# (resolved to a classifier/regressor variant once the task type is known),
# except for the inherently task-specific ones.
MODEL_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("xgboost", "xgboost"),
    ("xgb", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("lgbm", "lightgbm"),
    ("gradient boosted", "gradient_boosting"),
    ("gradient boosting", "gradient_boosting"),
    ("random forest", "random_forest"),
    ("decision tree", "decision_tree"),
    ("k-nearest neighbor", "knn"),
    ("nearest neighbor", "knn"),
    ("knn", "knn"),
    ("logistic regression", "logistic_regression"),
    ("ridge regression", "ridge_regression"),
    ("lasso regression", "lasso_regression"),
    ("linear regression", "linear_regression"),
]


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
                    actual_unique = pd.to_numeric(df[actual_cols[0]], errors="coerce").nunique(dropna=True)
                    pred_unique = pd.to_numeric(df[pred_cols[0]], errors="coerce").nunique(dropna=True)
                    if actual_unique <= 20 and pred_unique <= 20:
                        return "classification"
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

    def _extract_known_column(
        self, message: str, df: pd.DataFrame | None, extra_markers: tuple[str, ...] = ()
    ) -> str | None:
        """Find a column name the user named explicitly.

        "column <name>" is always treated as an explicit marker. Bare marker
        words like "in"/"for"/"predict" are too common in natural phrasing
        ("check for X imbalance") to trust without confirming the captured
        word is an actual column in the dataset.
        """
        lower_cols = {str(c).lower(): str(c) for c in df.columns} if df is not None else {}

        explicit = re.search(r"\bcolumn\s+([a-zA-Z0-9_]+)", message, flags=re.IGNORECASE)
        if explicit:
            token = explicit.group(1)
            return lower_cols.get(token.lower(), token)

        markers = ("in",) + extra_markers
        marker_pattern = "|".join(re.escape(marker) for marker in markers)
        candidate = re.search(rf"\b(?:{marker_pattern})\s+([a-zA-Z0-9_]+)\b", message, flags=re.IGNORECASE)
        if candidate and candidate.group(1).lower() in lower_cols:
            return lower_cols[candidate.group(1).lower()]

        return None

    def _rule_plan(
        self,
        message: str,
        dataset_id: str | None,
        df: pd.DataFrame | None,
        trained_model_ids: list[str] | None = None,
    ) -> list[ToolCall]:
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

        if any(k in m for k in [
            "insight", "key finding", "what stands out", "what's interesting",
            "tell me what's interesting", "surprising",
        ]):
            calls.append(ToolCall(name="auto_insights", arguments={}))

        if any(k in m for k in [
            "correlation", "correlated", "relationship between", "related to",
            "associated with", "association between",
        ]):
            calls.append(ToolCall(name="correlation_analysis", arguments={}))

        if any(k in m for k in [
            "trend", "over time", "month over month", "week over week",
            "seasonality", "time series", "growth rate", "period over period",
        ]):
            calls.append(ToolCall(name="trend_analysis", arguments={}))

        if any(k in m for k in ["profile", "summary", "overview", "columns", "schema"]):
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        if any(k in m for k in ["quality", "data quality", "diagnostic", "diagnostics", "healthcheck", "health check"]):
            calls.append(ToolCall(name="data_quality_report", arguments={"sample": 10000}))

        # Word-boundary match: "nan" as a plain substring also matches inside
        # unrelated words like "dominant", causing a spurious missingness call.
        if re.search(r"\b(missing|null|nan|incomplete)\b", m):
            calls.append(ToolCall(name="missingness_matrix", arguments={"top_n": 30}))

        if any(k in m for k in ["skew", "skewed", "long tail", "heavy tail"]):
            calls.append(ToolCall(name="skewed_features", arguments={"sample": 10000, "threshold": 1.0}))

        if any(k in m for k in ["overrepresented", "over-represented", "dominant", "imbalance", "class imbalance", "bias"]):
            matched_col = self._extract_known_column(message, df)
            if matched_col:
                calls.append(
                    ToolCall(
                        name="overrepresented_categories",
                        arguments={"col": matched_col, "threshold": 0.5, "top_k": 10},
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

        named_model_type = next((model_type for phrase, model_type in MODEL_TYPE_KEYWORDS if phrase in m), None)
        train_requested = any(k in m for k in [
            "train a model",
            "train model",
            "build a model",
            "fit a model",
            "train a classifier",
            "build a classifier",
            "train a regressor",
            "build a regressor",
            "build a predictor",
            "supervised learning",
        ]) or (
            named_model_type is not None and any(verb in m for verb in ["train", "build", "fit"])
        )
        if train_requested and df is not None:
            target_col = self._extract_known_column(message, df, extra_markers=("predict", "target", "for"))
            if target_col:
                arguments: dict = {"target_col": target_col}
                if named_model_type:
                    arguments["model_type"] = named_model_type
                calls.append(ToolCall(name="train_supervised_model", arguments=arguments))
            # If no target column can be confidently identified, skip rather
            # than guess — an incorrect target trains a meaningless model.

        score_requested = any(k in m for k in ["score with model", "apply model", "use model", "score model"])
        if score_requested:
            model_id_match = re.search(
                r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                message,
                flags=re.IGNORECASE,
            )
            if model_id_match:
                calls.append(
                    ToolCall(name="score_with_model", arguments={"model_id": model_id_match.group(1)})
                )
            elif trained_model_ids:
                # No explicit id, but a model was trained earlier in this
                # conversation ("score with the model you just trained") -
                # reuse the most recently trained one.
                calls.append(
                    ToolCall(name="score_with_model", arguments={"model_id": trained_model_ids[-1]})
                )
            # No model_id named or known from history: skip rather than guess.

        if not calls and df is not None:
            ml_task_hint = self._detect_ml_dataset(df)
            if ml_task_hint:
                calls.append(
                    ToolCall(
                        name="evaluate_ml_predictions",
                        arguments={"task_hint": ml_task_hint},
                    )
                )

        if not calls and dataset_id:
            # No specific tool matched: run the broad auto-insights sweep
            # rather than a bare profile, so an ambiguous question still
            # surfaces quality, relationship, anomaly, and trend findings.
            calls.append(ToolCall(name="auto_insights", arguments={}))

        return calls

    def plan(
        self,
        message: str,
        dataset_id: str | None,
        top_k: int = 6,
        conversation_history: list[dict[str, str]] | None = None,
        trained_model_ids: list[str] | None = None,
    ) -> tuple[list[ToolCall], list[dict], str, str | None, list[str]]:
        """Returns (tool_calls, citations, planning_source, llm_error, llm_notes)."""
        df = self._load_dataset_sample(dataset_id)

        enable_rag = os.getenv("ENABLE_RAG", "1") == "1"
        citations = self._get_rag().retrieve(message, top_k=top_k) if enable_rag else []

        if self.llm.enabled:
            try:
                calls, notes = self.llm.plan(
                    message,
                    dataset_id=dataset_id,
                    df=df,
                    citations=citations,
                    conversation_history=conversation_history,
                    trained_model_ids=trained_model_ids,
                )
                if calls:
                    return calls, citations, "llm", None, notes
                calls = self._rule_plan(message, dataset_id, df, trained_model_ids)
                return calls, citations, "rules", None, notes
            except LLMUnavailable as e:
                calls = self._rule_plan(message, dataset_id, df, trained_model_ids)
                return calls, citations, "rules", str(e), []

        calls = self._rule_plan(message, dataset_id, df, trained_model_ids)
        return calls, citations, "rules", None, []
