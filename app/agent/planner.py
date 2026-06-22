from __future__ import annotations

import os
import re

import pandas as pd

from app.analytics.dataset_manager import DatasetManager
from app.analytics.ml_train.model_store import ModelManager
from app.core.models import ToolCall
from app.agent.llm import LLMReasoner, LLMUnavailable, LATEST_TRAINED_MODEL_SENTINEL
from app.agent.llm_metrics import metrics

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
        self.model_manager = ModelManager()
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
        # Iterate all matches: the first hit may capture a non-column word
        # (e.g. "for me on debit" → "for" captures "me" before "on" captures "debit").
        for m in re.finditer(rf"\b(?:{marker_pattern})\s+([a-zA-Z0-9_]+)\b", message, flags=re.IGNORECASE):
            if m.group(1).lower() in lower_cols:
                return lower_cols[m.group(1).lower()]

        # Bare column name — the entire message IS the column name.
        stripped = message.strip()
        if stripped.lower() in lower_cols:
            return lower_cols[stripped.lower()]

        # Unambiguous prefix match for short replies (e.g. "A" → "amount").
        # Only fires when exactly one column starts with the prefix, so "ba" → "balance"
        # works but "b" stays ambiguous if "balance" and "balance_usd" both exist.
        stripped_lower = stripped.lower()
        if 1 <= len(stripped_lower) <= 6 and re.fullmatch(r"[a-z0-9_]+", stripped_lower):
            prefix_matches = [col for col_lower, col in lower_cols.items()
                              if col_lower.startswith(stripped_lower)]
            if len(prefix_matches) == 1:
                return prefix_matches[0]

        return None

    def _rule_plan(
        self,
        message: str,
        dataset_id: str | None,
        df: pd.DataFrame | None,
        trained_model_ids: list[str] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> list[ToolCall]:
        m = (message or "").lower()
        calls: list[ToolCall] = []

        ml_requested = any(k in m for k in [
            "evaluate_ml_predictions",
            "ml evaluation",
            "model evaluation",
            "evaluate model",
            "evaluate this model",
            "evaluate the model",
            "evaluate trained model",
            "evaluate latest model",
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
            trained_eval_call = self._trained_model_eval_call(message, trained_model_ids, dataset_id)
            if trained_eval_call is not None:
                calls.append(trained_eval_call)
            else:
                task_hint = self._detect_ml_dataset(df) or "auto"
                calls.append(
                    ToolCall(
                        name="evaluate_ml_predictions",
                        arguments={"task_hint": task_hint},
                    )
                )

        if any(k in m for k in [
            "analyse", "analyze", "full analysis", "deep dive", "comprehensive",
            "understand this data", "tell me about this", "what can you tell me",
            "analyse this", "analyze this", "analyse the data", "analyze the data",
        ]):
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))
            calls.append(ToolCall(name="data_quality_report", arguments={"sample": 10000}))
            calls.append(ToolCall(name="auto_insights", arguments={}))
            calls.append(ToolCall(name="correlation_analysis", arguments={}))

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

        if any(k in m for k in ["profile", "summary", "summarize", "overview", "columns", "schema", "descriptive statistics", "describe the"]):
            calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

        if any(k in m for k in ["quality", "data quality", "diagnostic", "diagnostics", "healthcheck", "health check", "duplicate", "duplicated rows"]):
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

        if "pivot" in m or "group by" in m or "breakdown" in m or "break down" in m or "broken down" in m:
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
            idx = m.index("sql:") + 4
            q = message[idx:].strip()
            calls.append(ToolCall(name="duckdb_query", arguments={"query": q}))

        if "anomal" in m or "outlier" in m:
            calls.append(ToolCall(name="anomaly_scan", arguments={"numeric_cols": [], "contamination": 0.02}))

        if "cluster" in m or "segment" in m or "grouping" in m or "natural group" in m:
            calls.append(ToolCall(name="kmeans_clusters", arguments={"numeric_cols": [], "k": 5}))

        named_model_type = next((model_type for phrase, model_type in MODEL_TYPE_KEYWORDS if phrase in m), None)
        train_requested = any(k in m for k in [
            "train a model",
            "train model",
            "build a model",
            "fit a model",
            "train a classifier",
            "build a classifier",
            "fit a classifier",
            "train a regressor",
            "build a regressor",
            "fit a regressor",
            "build a predictor",
            "build a regression",
            "build a classification",
            "supervised learning",
        ]) or (
            named_model_type is not None and any(verb in m for verb in ["train", "build", "fit"])
        )
        if train_requested and df is not None:
            target_col = self._extract_known_column(message, df, extra_markers=("predict", "target", "for", "on"))
            if target_col:
                arguments: dict = {"target_col": target_col}
                if named_model_type:
                    arguments["model_type"] = named_model_type
                calls.append(ToolCall(name="train_supervised_model", arguments=arguments))
            else:
                # No target column named — profile the dataset so the user can
                # see which columns are available and pick one.
                calls.append(ToolCall(name="profile_dataset", arguments={"sample": 5000}))

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

        explain_requested = any(k in m for k in [
            "explain model", "explain the model", "feature importance",
            "why does the model", "what features", "feature contribution",
            "which features matter", "model explainability", "permutation importance",
            "shap explanation", "shap explain", "shap values", "shap importance",
            "show shap", "give shap",
        ])
        if explain_requested:
            model_id_match = re.search(
                r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                message,
                flags=re.IGNORECASE,
            )
            if model_id_match:
                calls.append(
                    ToolCall(name="explain_model", arguments={"model_id": model_id_match.group(1)})
                )
            elif trained_model_ids:
                calls.append(
                    ToolCall(name="explain_model", arguments={"model_id": trained_model_ids[-1]})
                )
            elif train_requested:
                # Train + explain in the same request: use sentinel — executor resolves
                # the real model_id once train_supervised_model finishes.
                calls.append(
                    ToolCall(name="explain_model", arguments={"model_id": LATEST_TRAINED_MODEL_SENTINEL})
                )

        # Per-prediction local SHAP explanation
        local_explain_requested = any(k in m for k in [
            "why did the model predict", "explain this prediction", "explain prediction",
            "what drove", "local explanation", "why was this predicted",
            "explain row", "prediction for row",
            "shap prediction", "shap row", "local shap",
        ])
        if local_explain_requested:
            model_id_match = re.search(
                r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                message,
                flags=re.IGNORECASE,
            )
            row_match = re.search(r"\brow[_\s]?(\d+)\b", message, flags=re.IGNORECASE)
            row_idx = int(row_match.group(1)) if row_match else 0
            if model_id_match:
                calls.append(ToolCall(name="shap_explain_prediction", arguments={"model_id": model_id_match.group(1), "row_idx": row_idx}))
            elif trained_model_ids:
                calls.append(ToolCall(name="shap_explain_prediction", arguments={"model_id": trained_model_ids[-1], "row_idx": row_idx}))

        # Time-series forecast
        forecast_requested = any(k in m for k in [
            "forecast", "predict next", "future values", "next n days",
            "next few days", "next week", "next month", "project ahead",
            "project forward", "extrapolate", "predict future",
        ])
        if forecast_requested:
            model_id_match = re.search(
                r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                message,
                flags=re.IGNORECASE,
            )
            horizon_match = re.search(r"\b(\d+)\s*(day|week|month|step)s?\b", message, flags=re.IGNORECASE)
            horizon = 30
            if horizon_match:
                n = int(horizon_match.group(1))
                unit = horizon_match.group(2).lower()
                horizon = n * (7 if unit == "week" else 30 if unit == "month" else 1)
                horizon = min(max(horizon, 1), 365)
            if model_id_match:
                calls.append(ToolCall(name="forecast_with_model", arguments={"model_id": model_id_match.group(1), "horizon": horizon}))
            elif trained_model_ids:
                calls.append(ToolCall(name="forecast_with_model", arguments={"model_id": trained_model_ids[-1], "horizon": horizon}))

        if not calls and df is not None:
            ml_task_hint = self._detect_ml_dataset(df)
            if ml_task_hint:
                calls.append(
                    ToolCall(
                        name="evaluate_ml_predictions",
                        arguments={"task_hint": ml_task_hint},
                    )
                )

        # Follow-up: user replied with just a column name after the assistant
        # asked which column to predict.  Check conversation history for a
        # pending training request.
        if not calls and df is not None and conversation_history:
            bare_col = self._extract_known_column(message, df)
            if bare_col:
                last_assistant = next(
                    (turn.get("content", "") for turn in reversed(conversation_history)
                     if turn.get("role") == "assistant"),
                    "",
                )
                _train_follow_up_signals = [
                    "which column to predict", "specify the target",
                    "train a model", "need to know", "build a model",
                ]
                if any(sig in last_assistant.lower() for sig in _train_follow_up_signals):
                    calls.append(ToolCall(name="train_supervised_model", arguments={"target_col": bare_col}))

        if not calls and dataset_id:
            # No specific tool matched: run the broad auto-insights sweep
            # rather than a bare profile, so an ambiguous question still
            # surfaces quality, relationship, anomaly, and trend findings.
            metrics.record_fallback("no_tool_matched")
            calls.append(ToolCall(name="auto_insights", arguments={}))

        return calls

    def _trained_model_eval_call(
        self,
        message: str,
        trained_model_ids: list[str] | None = None,
        dataset_id: str | None = None,
    ) -> ToolCall | None:
        m = (message or "").lower()
        if not any(k in m for k in [
            "evaluate model",
            "evaluate this model",
            "evaluate the model",
            "evaluate trained model",
            "evaluate latest model",
            "model evaluation",
        ]):
            return None

        model_id_match = re.search(
            r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
            message,
            flags=re.IGNORECASE,
        )
        if model_id_match:
            return ToolCall(name="evaluate_trained_model", arguments={"model_id": model_id_match.group(1)})

        refers_to_known_model = any(k in m for k in ["this model", "the model", "trained model", "latest model"])
        if trained_model_ids and refers_to_known_model:
            return ToolCall(name="evaluate_trained_model", arguments={"model_id": trained_model_ids[-1]})

        if dataset_id and refers_to_known_model:
            try:
                models = self.model_manager.list_models()
            except Exception:
                models = []
            candidates = [model for model in models if model.dataset_id == dataset_id]
            if candidates:
                latest = max(candidates, key=lambda model: model.created_at)
                return ToolCall(name="evaluate_trained_model", arguments={"model_id": latest.model_id})

        return None

    def plan(
        self,
        message: str,
        dataset_id: str | None,
        top_k: int = 6,
        conversation_history: list[dict] | None = None,
        trained_model_ids: list[str] | None = None,
    ) -> tuple[list[ToolCall], list[dict], str, str | None, list[str]]:
        """Returns (tool_calls, citations, planning_source, llm_error, llm_notes)."""
        df = self._load_dataset_sample(dataset_id)

        enable_rag = os.getenv("ENABLE_RAG", "1") == "1"
        citations = self._get_rag().retrieve(message, top_k=top_k) if enable_rag else []

        trained_eval_call = self._trained_model_eval_call(message, trained_model_ids, dataset_id)
        if trained_eval_call is not None and df is not None:
            return [trained_eval_call], citations, "rules", None, []

        m = (message or "").lower()
        deterministic_explain_requested = any(k in m for k in [
            "explain model", "explain the model", "feature importance",
            "why does the model", "what features", "feature contribution",
            "which features matter", "model explainability", "permutation importance",
            "shap explanation", "shap explain", "shap values", "shap importance",
            "show shap", "give shap", "why did the model predict",
            "explain this prediction", "explain prediction", "what drove",
            "local explanation", "why was this predicted", "explain row",
            "prediction for row", "shap prediction", "shap row", "local shap",
        ])
        if deterministic_explain_requested and df is not None:
            calls = self._rule_plan(message, dataset_id, df, trained_model_ids, conversation_history)
            if any(call.name in {"explain_model", "shap_explain_prediction"} for call in calls):
                return calls, citations, "rules", None, []

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
                calls = self._rule_plan(message, dataset_id, df, trained_model_ids, conversation_history)
                return calls, citations, "rules", None, notes
            except LLMUnavailable as e:
                calls = self._rule_plan(message, dataset_id, df, trained_model_ids, conversation_history)
                return calls, citations, "rules", str(e), []

        calls = self._rule_plan(message, dataset_id, df, trained_model_ids, conversation_history)
        return calls, citations, "rules", None, []
