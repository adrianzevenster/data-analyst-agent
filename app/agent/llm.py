from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from pydantic import ValidationError

from app.agent.llm_metrics import LLMCallRecord, RepairRecord, metrics
from app.analytics.relationships import correlation_ratio
from app.analytics.tooling import get_registry
from app.analytics.tool_validation import format_validation_error, validate_tool_args
from app.analytics.trends import detect_datetime_col, trend_analysis
from app.core.config import settings
from app.core.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)

OUTLIER_Z_SCORE_THRESHOLD = 3.0
MAX_CATEGORICAL_ASSOCIATION_CARDINALITY = 20

# Placeholder model_id the planner emits when a score_with_model call should
# use a model trained earlier in the *same* plan (so the real id doesn't
# exist yet at planning time - training hasn't run). The executor resolves
# this against that batch's own tool_results before running the call.
LATEST_TRAINED_MODEL_SENTINEL = "<latest_trained_model_id>"

class LLMUnavailable(RuntimeError):
    pass


class LLMReasoner:
    """
    OpenAI-compatible client for local/open-weight inference servers.

    Recommended serving targets:
    - Qwen/Qwen3-32B for a strong single-GPU reasoning baseline.
    - Qwen/Qwen3-235B-A22B or DeepSeek-R1 class models for larger GPU fleets.
    - Mistral Small 3.2 when tool-calling reliability matters more than deep reasoning.
    """

    def __init__(self) -> None:
        self.registry = get_registry()

    @property
    def enabled(self) -> bool:
        return bool(settings.llm_enabled and settings.llm_base_url and settings.llm_model)

    def _chat(
        self, messages: list[dict[str, str]], *, temperature: float | None = None, operation: str = "chat"
    ) -> str:
        if not self.enabled:
            raise LLMUnavailable("LLM inference is disabled or not configured.")

        url = (settings.llm_base_url or "").rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        payload: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if settings.llm_json_mode and operation in ("plan", "repair", "judge"):
            payload["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=settings.llm_timeout_seconds)
            resp.raise_for_status()
            body = resp.json()
            latency_ms = (time.monotonic() - start) * 1000
            usage = body.get("usage") or {}
            total_tokens = usage.get("total_tokens")
            metrics.record(
                LLMCallRecord(operation=operation, ok=True, latency_ms=latency_ms, total_tokens=total_tokens)
            )
            logger.info(
                "llm_call operation=%s ok=true latency_ms=%.1f total_tokens=%s",
                operation, latency_ms, total_tokens,
            )
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            metrics.record(LLMCallRecord(operation=operation, ok=False, latency_ms=latency_ms, error=str(exc)))
            logger.warning("llm_call operation=%s ok=false latency_ms=%.1f error=%s", operation, latency_ms, exc)
            raise LLMUnavailable(f"LLM inference request failed: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(stripped[start : end + 1])

    @staticmethod
    def _json_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    @classmethod
    def rag_context(cls, citations: list[dict]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for citation in citations[: settings.llm_rag_max_chunks]:
            text = str(citation.get("text", "")).strip()
            if not text:
                continue
            context.append(
                {
                    "source_id": citation.get("source_id"),
                    "score": cls._json_value(citation.get("score")),
                    "text": text[: settings.llm_rag_max_chars_per_chunk],
                }
            )
        return context

    @classmethod
    def _planning_dataset_context(cls, df: pd.DataFrame | None) -> dict[str, Any] | None:
        if df is None:
            return None

        sample = df.head(settings.llm_analysis_preview_rows)
        return {
            "rows_sampled": int(df.shape[0]),
            "columns": [
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "missing_pct": round(float(df[col].isna().mean() * 100), 3),
                }
                for col in df.columns[: settings.llm_analysis_max_columns]
            ],
            "sample_rows": sample.map(cls._json_value).to_dict(orient="records"),
        }

    @staticmethod
    def _outlier_count(series: pd.Series) -> int | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        std = clean.std()
        if len(clean) < 3 or not std:
            return None
        z_scores = (clean - clean.mean()).abs() / std
        return int((z_scores >= OUTLIER_Z_SCORE_THRESHOLD).sum())

    @classmethod
    def dataset_analysis_context(cls, df: pd.DataFrame | None) -> dict[str, Any] | None:
        if df is None:
            return None

        limited = df.iloc[:, : settings.llm_analysis_max_columns]
        numeric_cols = [
            c for c in limited.columns
            if pd.api.types.is_numeric_dtype(limited[c]) and not pd.api.types.is_bool_dtype(limited[c])
        ]
        numeric = limited[numeric_cols]
        categorical_cols = [
            c for c in limited.columns
            if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(limited[c])
        ]

        column_profiles: list[dict[str, Any]] = []
        for col in limited.columns:
            series = limited[col]
            profile: dict[str, Any] = {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 3),
                "unique_count": int(series.nunique(dropna=True)),
            }

            if col in numeric_cols:
                described = series.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                profile["numeric_summary"] = {
                    str(k): cls._json_value(round(float(v), 6))
                    for k, v in described.items()
                    if pd.notna(v)
                }
                outliers = cls._outlier_count(series)
                if outliers is not None:
                    profile["outlier_count_zscore_3"] = outliers
            else:
                top_values = series.astype("string").value_counts(dropna=True).head(5)
                profile["top_values"] = [
                    {"value": cls._json_value(value), "count": int(count)}
                    for value, count in top_values.items()
                ]

            column_profiles.append(profile)

        correlations: list[dict[str, Any]] = []
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True).abs()
            seen: set[tuple[str, str]] = set()
            for left in corr.columns:
                for right in corr.columns:
                    if left == right:
                        continue
                    pair: tuple[str, str] = (min(str(left), str(right)), max(str(left), str(right)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    value = corr.loc[left, right]
                    if pd.notna(value):
                        correlations.append(
                            {"columns": [str(left), str(right)], "abs_correlation": round(float(value), 6)}
                        )
            correlations.sort(key=lambda item: item["abs_correlation"], reverse=True)

        categorical_associations: list[dict[str, Any]] = []
        for cat_col in categorical_cols:
            nunique = limited[cat_col].nunique(dropna=True)
            if not (2 <= nunique <= MAX_CATEGORICAL_ASSOCIATION_CARDINALITY):
                continue
            for num_col in numeric_cols:
                ratio = correlation_ratio(limited[cat_col], limited[num_col])
                if ratio is not None:
                    categorical_associations.append(
                        {"categorical_column": str(cat_col), "numeric_column": str(num_col), "correlation_ratio": round(ratio, 6)}
                    )
        categorical_associations.sort(key=lambda item: item["correlation_ratio"], reverse=True)

        trend_summary: dict[str, Any] | None = None
        date_col = detect_datetime_col(limited)
        if date_col:
            trend = trend_analysis(limited, date_col=date_col)
            if "error" not in trend:
                trend_summary = {
                    "date_col": trend["date_col"],
                    "value_col": trend["value_col"],
                    "direction": trend["direction"],
                    "overall_change_pct": trend["overall_change_pct"],
                    "engineering_readout": trend["engineering_readout"],
                }

        sample = limited.head(settings.llm_analysis_preview_rows)
        return {
            "rows_sampled": int(df.shape[0]),
            "columns_sampled": int(limited.shape[1]),
            "column_profiles": column_profiles,
            "strongest_numeric_correlations": correlations[: settings.llm_analysis_max_correlations],
            "strongest_categorical_associations": categorical_associations[
                : settings.llm_analysis_max_categorical_associations
            ],
            "trend_summary": trend_summary,
            "sample_rows": sample.map(cls._json_value).to_dict(orient="records"),
        }

    _FEW_SHOT_EXAMPLES: list[dict] = [
        {
            "message": "Give me a profile of this dataset",
            "tool_calls": [{"name": "profile_dataset", "arguments": {"sample": 5000}}],
        },
        {
            "message": "Run a data quality healthcheck",
            "tool_calls": [{"name": "data_quality_report", "arguments": {"sample": 10000}}],
        },
        {
            "message": "Are any features skewed?",
            "tool_calls": [{"name": "skewed_features", "arguments": {"threshold": 1.0}}],
        },
        {
            "message": "Find outliers in this data",
            "tool_calls": [{"name": "anomaly_scan", "arguments": {"contamination": 0.02}}],
        },
        {
            "message": "Cluster these rows into segments",
            "tool_calls": [{"name": "kmeans_clusters", "arguments": {"k": 5}}],
        },
        {
            "message": "What correlations exist in this data?",
            "tool_calls": [{"name": "correlation_analysis", "arguments": {}}],
        },
        {
            "message": "Show me the trend over time",
            "tool_calls": [{"name": "trend_analysis", "arguments": {}}],
        },
        {
            "message": "What insights stand out in this data?",
            "tool_calls": [{"name": "auto_insights", "arguments": {}}],
        },
        {
            "message": "Analyse this for me",
            "tool_calls": [
                {"name": "profile_dataset", "arguments": {"sample": 5000}},
                {"name": "data_quality_report", "arguments": {"sample": 10000}},
                {"name": "auto_insights", "arguments": {}},
                {"name": "correlation_analysis", "arguments": {}},
            ],
        },
        {
            "message": "Give me a breakdown by region",
            "tool_calls": [{"name": "multidim_pivot", "arguments": {"index": ["region"], "agg": "sum"}}],
        },
        {
            "message": "Train a random forest to predict revenue",
            "tool_calls": [{"name": "train_supervised_model", "arguments": {"target_col": "revenue", "model_type": "random_forest"}}],
        },
        {
            "message": "Train a model for me",
            "tool_calls": [{"name": "profile_dataset", "arguments": {"sample": 5000}}],
        },
        {
            "message": "How accurate are the churn predictions?",
            "tool_calls": [{"name": "evaluate_ml_predictions", "arguments": {"task_hint": "classification"}}],
        },
        {
            "message": "Which columns have missing values?",
            "tool_calls": [
                {"name": "missingness_matrix", "arguments": {"top_n": 20}},
                {"name": "profile_dataset", "arguments": {"sample": 5000}},
            ],
        },
        {
            "message": "What is the total revenue per region?",
            "tool_calls": [
                {
                    "name": "duckdb_query",
                    "arguments": {"query": "SELECT region, SUM(revenue) AS total_revenue FROM t GROUP BY region ORDER BY total_revenue DESC"},
                }
            ],
        },
        {
            "message": "Show me the top 5 customers by sales amount",
            "tool_calls": [
                {
                    "name": "duckdb_query",
                    "arguments": {"query": "SELECT customer, SUM(sales_amount) AS total_sales FROM t GROUP BY customer ORDER BY total_sales DESC LIMIT 5"},
                }
            ],
        },
        {
            "message": "How many orders were placed each month?",
            "tool_calls": [
                {
                    "name": "duckdb_query",
                    "arguments": {"query": "SELECT strftime(order_date, '%Y-%m') AS month, COUNT(*) AS orders FROM t GROUP BY month ORDER BY month"},
                }
            ],
        },
        {
            "message": "Forecast sales for the next 30 days",
            "tool_calls": [{"name": "forecast_with_model", "arguments": {"model_id": "<latest_trained_model_id>", "horizon": 30}}],
        },
        {
            "message": "Why did the model predict churn for row 5?",
            "tool_calls": [{"name": "shap_explain_prediction", "arguments": {"model_id": "<latest_trained_model_id>", "row_idx": 5}}],
        },
        {
            "message": "Train a model to predict revenue and explain the important features",
            "tool_calls": [
                {"name": "train_supervised_model", "arguments": {"target_col": "revenue"}},
                {"name": "explain_model", "arguments": {"model_id": "<latest_trained_model_id>"}},
            ],
        },
    ]

    _PLANNER_SYSTEM_PROMPT = (
        "You are a staff-level data analyst agent planner. "
        "Select the smallest useful set of analytics tools. "
        "Return strict JSON only — no prose, no explanation, no markdown fences — "
        "with shape {\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}. "
        "Use rag_context for domain guidance when it is relevant. Use only tools "
        "from available_tools. Do not invent columns; leave ambiguous tool "
        "arguments empty so the executor can infer safe defaults. "
        "If the user names a specific model family or algorithm for "
        "train_supervised_model (e.g. random forest, xgboost, lightgbm, "
        "gradient boosting, decision tree, knn, ridge, lasso, logistic "
        "regression, linear regression), set model_type to that exact name "
        "from the tool's schema - never leave it as \"auto\" when one is named. "
        "Use conversation_history to resolve references to earlier turns "
        "(e.g. \"that dataset\", \"run it again\", \"train another one\"). "
        "For evaluate_trained_model: if the user asks to evaluate this/the/latest "
        "trained model and known_trained_model_ids is non-empty, use the exact "
        "last id string from known_trained_model_ids as model_id. Do not turn "
        "trained-model evaluation into SQL. "
        "For score_with_model: if the user refers to a model trained earlier "
        "in conversation_history and known_trained_model_ids is non-empty, "
        "use the exact id string from known_trained_model_ids as model_id - "
        f"never write a description, placeholder, or instructions as the id. "
        f"If the user asks to train and then immediately score in this same "
        f"request, the new model's id does not exist yet; set model_id to the "
        f"exact literal string {LATEST_TRAINED_MODEL_SENTINEL!r} for that call "
        f"and the executor will substitute the real id once training runs. "
        "Natural language aggregation and filtering: when the user asks for a "
        "breakdown, summary, count, total, average, ranking, or filter that maps "
        "naturally to SQL (e.g. 'total revenue by region', 'top 5 customers', "
        "'orders per month', 'how many X have Y'), prefer duckdb_query with a "
        "concise SQL query. The active dataset is always registered as table 't'. "
        "Write only standard SQL that DuckDB supports; do NOT use Python or "
        "pandas syntax inside the query string. "
        "Study the few_shot_examples to learn the expected output format and "
        "tool selection patterns before planning. "
        "For forecast_with_model: only use when the user explicitly asks to forecast or predict future values; "
        "requires a model_id from a temporal regression model. "
        "For shap_explain_prediction: use when the user wants to understand why the model made a specific prediction "
        "for a single row; requires model_id and row_idx."
    )

    @staticmethod
    def _parse_tool_calls(content: str, known: set[str]) -> list[ToolCall]:
        parsed = LLMReasoner._extract_json(content)
        calls: list[ToolCall] = []
        for raw_call in parsed.get("tool_calls", []):
            if not isinstance(raw_call, dict):
                continue
            name = raw_call.get("name")
            if name not in known:
                continue
            args = raw_call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=name, arguments=args))
        return calls

    @staticmethod
    def _validate_calls(
        calls: list[ToolCall], df: pd.DataFrame
    ) -> tuple[list[ToolCall], list[dict[str, Any]]]:
        """Schema-ground each call against the real dataset.

        Returns (valid_calls, problems) where problems carries enough detail
        (call name/arguments/error) for a repair prompt to fix just the
        offending calls.
        """
        registry = get_registry()
        valid: list[ToolCall] = []
        problems: list[dict[str, Any]] = []
        for call in calls:
            try:
                tool = registry.get(call.name)
                args = tool.validate_args(call.arguments)
                validate_tool_args(df, call.name, args)
                valid.append(ToolCall(name=call.name, arguments=args))
            except ValidationError as exc:
                problems.append(
                    {"name": call.name, "arguments": call.arguments, "error": format_validation_error(exc)}
                )
            except (ValueError, KeyError) as exc:
                problems.append({"name": call.name, "arguments": call.arguments, "error": str(exc)})
        return valid, problems

    def _repair_calls(
        self,
        problems: list[dict[str, Any]],
        *,
        message: str,
        dataset_context: dict[str, Any] | None,
        citations: list[dict],
        tools: list[dict],
        known: set[str],
    ) -> list[ToolCall]:
        """One repair round-trip: hand the LLM its own invalid calls plus the
        validation errors and the real schema, and ask it to fix just those.
        """
        repair_payload = {
            "user_message": message,
            "dataset": dataset_context,
            "rag_context": self.rag_context(citations),
            "available_tools": tools,
            "invalid_tool_calls": problems,
        }

        content = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "The tool calls below failed validation against the real dataset "
                        "schema (unknown columns or bad arguments). Fix each one using only "
                        "columns listed in dataset.columns and the schemas in available_tools. "
                        "If a call cannot be fixed, drop it. Return strict JSON only with shape "
                        "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}."
                    ),
                },
                {"role": "user", "content": json.dumps(repair_payload, default=str)},
            ],
            temperature=0.0,
            operation="repair",
        )

        try:
            return self._parse_tool_calls(content, known)
        except Exception:
            return []

    def plan(
        self,
        message: str,
        *,
        dataset_id: str | None,
        df: pd.DataFrame | None,
        citations: list[dict],
        conversation_history: list[dict[str, str]] | None = None,
        trained_model_ids: list[str] | None = None,
    ) -> tuple[list[ToolCall], list[str]]:
        """Returns (tool_calls, notes). Notes record any calls the LLM
        proposed that didn't survive schema validation/repair, so callers can
        surface what was dropped instead of silently losing it.
        """
        tools = self.registry.list()
        known = {t["name"] for t in tools}
        dataset_context = self._planning_dataset_context(df)
        prompt_payload = {
            "user_message": message,
            "dataset_id": dataset_id,
            "dataset": dataset_context,
            "rag_context": self.rag_context(citations),
            "available_tools": tools,
            "conversation_history": conversation_history or [],
            "known_trained_model_ids": trained_model_ids or [],
            "few_shot_examples": self._FEW_SHOT_EXAMPLES,
        }

        content = self._chat(
            [
                {"role": "system", "content": self._PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt_payload, default=str)},
            ],
            temperature=0.0,
            operation="plan",
        )

        try:
            calls = self._parse_tool_calls(content, known)
        except Exception as exc:
            raise LLMUnavailable(f"LLM planner returned invalid tool JSON: {exc}") from exc

        if df is None:
            return calls[: settings.llm_max_tool_calls], []

        valid, problems = self._validate_calls(calls, df)
        # Track every call that has ever failed validation, even if a repair
        # round-trip simply drops it (rather than fixing it) - the caller
        # still needs to know it was requested and discarded.
        dropped: dict[str, str] = {p["name"]: p["error"] for p in problems}

        n_initial_problems = len(problems)
        attempts = 0
        while problems and attempts < settings.llm_max_repair_attempts:
            attempts += 1
            try:
                repaired = self._repair_calls(
                    problems,
                    message=message,
                    dataset_context=dataset_context,
                    citations=citations,
                    tools=tools,
                    known=known,
                )
            except LLMUnavailable:
                break
            repaired_valid, problems = self._validate_calls(repaired, df)
            for call in repaired_valid:
                dropped.pop(call.name, None)
            for p in problems:
                dropped[p["name"]] = p["error"]
            valid.extend(repaired_valid)

        if attempts > 0:
            metrics.record_repair(
                RepairRecord(
                    n_problems_in=n_initial_problems,
                    n_fixed=n_initial_problems - len(problems),
                    n_dropped=len(problems),
                )
            )

        notes = [f"Dropped tool call '{name}' after validation failed: {error}" for name, error in dropped.items()]

        return valid[: settings.llm_max_tool_calls], notes

    @staticmethod
    def _slim_tool_results(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
        """Strip raw nested result blobs — keep only the human-readable summary fields.

        Open-weight models tend to describe JSON structure when they see large nested
        objects. Sending only the text fields they need to synthesise from prevents that.
        """
        slim = []
        for tr in tool_results:
            entry: dict[str, Any] = {"tool": tr.name, "ok": tr.ok}
            if not tr.ok:
                entry["error"] = tr.error
            elif isinstance(tr.result, dict):
                # Surface the pre-written readout if available; otherwise a compact summary.
                readout = tr.result.get("engineering_readout") or tr.result.get("readout")
                if readout:
                    entry["summary"] = str(readout)
                # Include short text lists (findings, notes) but skip nested dicts/charts.
                for key in ("findings", "notes", "warnings"):
                    val = tr.result.get(key)
                    if isinstance(val, list) and val and isinstance(val[0], str):
                        entry[key] = val[:10]
                # Scalar top-level metrics (row counts, scores, etc.)
                for key, val in tr.result.items():
                    if key in ("engineering_readout", "readout", "findings", "notes",
                               "warnings", "charts", "columns", "rows"):
                        continue
                    if isinstance(val, (int, float, str, bool)) and not isinstance(val, bool):
                        entry[key] = val
            slim.append(entry)
        return slim

    def synthesize(
        self,
        message: str,
        *,
        dataset_id: str | None,
        dataset_context: dict[str, Any] | None,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        citations: list[dict],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        payload = {
            "user_message": message,
            "dataset_id": dataset_id,
            "dataset_context": dataset_context,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "tool_results": self._slim_tool_results(tool_results),
            "rag_context": self.rag_context(citations),
            "conversation_history": conversation_history or [],
        }

        return self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise senior data analyst writing a response to a business user. "
                        "Your output is the final answer shown in the UI — write it as clear, "
                        "plain English prose or bullet points. "
                        "NEVER describe the structure of the JSON you received. NEVER say things like "
                        "'This is a JSON object', 'The results array contains', 'top-level keys', "
                        "or anything that references the payload structure. "
                        "Instead, read the data in tool_results and dataset_context and write actual "
                        "findings: specific numbers, column names, trends, anomalies, and "
                        "actionable observations drawn directly from the evidence. "
                        "Use rag_context for domain guidance when relevant, but do not quote it "
                        "verbatim or mention it by name. "
                        "Use conversation_history only to avoid re-explaining established facts. "
                        "Do not expose internal chain-of-thought. "
                        "If evidence is insufficient to answer, say what is needed next. "
                        "If the user asked to train a model but no train_supervised_model result "
                        "is present in tool_results, tell them clearly: list the available columns "
                        "from dataset_context and ask which one to predict — e.g. "
                        "'To train a model I need a target column. Your dataset has: col1, col2, "
                        "col3 — which would you like to predict?'"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=settings.llm_temperature,
            operation="synthesize",
        ).strip()

    def judge_groundedness(
        self,
        answer: str,
        *,
        dataset_context: dict[str, Any] | None,
        tool_results: list[ToolResult],
    ) -> dict[str, Any]:
        """LLM-as-judge: scores whether `answer` is actually supported by the
        evidence it was synthesized from, rather than fabricating claims.

        Returns {"score": 1-5, "issues": [...]}. Raises LLMUnavailable on
        request failure or unparseable judge output - callers decide whether
        a failed judgement should block anything (it shouldn't; it's a
        sampled quality signal, not a gate).
        """
        payload = {
            "answer_to_judge": answer,
            "dataset_context": dataset_context,
            "tool_results": [tr.model_dump() for tr in tool_results],
        }

        content = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact-checking judge. Given an analyst's answer and the "
                        "evidence it should be grounded in (dataset_context, tool_results), score "
                        "the answer on three criteria, each 1-5. Return strict JSON only:\n"
                        "{\n"
                        "  \"groundedness\": <1-5>,\n"
                        "  \"accuracy\": <1-5>,\n"
                        "  \"completeness\": <1-5>,\n"
                        "  \"unsupported_claims\": [\"...\"]\n"
                        "}\n"
                        "groundedness: every claim is directly traceable to the evidence "
                        "(5=fully, 1=mostly fabricated). "
                        "accuracy: numbers/statistics in the answer match the evidence exactly "
                        "(5=all correct, 1=mostly wrong). "
                        "completeness: the answer addresses the user's actual question "
                        "(5=fully, 1=ignores it). "
                        "unsupported_claims: list specific sentences or numbers that are NOT "
                        "backed by the evidence; empty list if none."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=0.0,
            operation="judge",
        )

        try:
            parsed = self._extract_json(content)
            def _clamp(val, default=3):
                try:
                    return max(1, min(5, int(val or default)))
                except (TypeError, ValueError):
                    return default
            criteria = {
                "groundedness": _clamp(parsed.get("groundedness")),
                "accuracy":     _clamp(parsed.get("accuracy")),
                "completeness": _clamp(parsed.get("completeness")),
            }
            overall = round(sum(criteria.values()) / len(criteria))
            issues = [str(x) for x in parsed.get("unsupported_claims", []) if isinstance(x, str)]
            return {"score": overall, "criteria": criteria, "issues": issues}
        except Exception as exc:
            raise LLMUnavailable(f"LLM judge returned invalid JSON: {exc}") from exc
