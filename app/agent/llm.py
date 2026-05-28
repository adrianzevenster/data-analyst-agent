from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import requests

from app.analytics.tooling import get_registry
from app.core.config import settings
from app.core.models import ToolCall, ToolResult

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

    def _chat(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        if not self.enabled:
            raise LLMUnavailable("LLM inference is disabled or not configured.")

        url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=settings.llm_timeout_seconds)
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
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

    @classmethod
    def dataset_analysis_context(cls, df: pd.DataFrame | None) -> dict[str, Any] | None:
        if df is None:
            return None

        limited = df.iloc[:, : settings.llm_analysis_max_columns]
        numeric = limited.select_dtypes(include="number")
        categorical = limited.select_dtypes(exclude="number")

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

            if pd.api.types.is_numeric_dtype(series):
                described = series.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                profile["numeric_summary"] = {
                    str(k): cls._json_value(round(float(v), 6))
                    for k, v in described.items()
                    if pd.notna(v)
                }
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
                    pair = tuple(sorted((str(left), str(right))))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    value = corr.loc[left, right]
                    if pd.notna(value):
                        correlations.append(
                            {"columns": [str(left), str(right)], "abs_correlation": round(float(value), 6)}
                        )
            correlations.sort(key=lambda item: item["abs_correlation"], reverse=True)

        sample = limited.head(settings.llm_analysis_preview_rows)
        return {
            "rows_sampled": int(df.shape[0]),
            "columns_sampled": int(limited.shape[1]),
            "column_profiles": column_profiles,
            "strongest_numeric_correlations": correlations[:10],
            "sample_rows": sample.map(cls._json_value).to_dict(orient="records"),
        }

    def plan(
        self,
        message: str,
        *,
        dataset_id: str | None,
        df: pd.DataFrame | None,
        citations: list[dict],
    ) -> list[ToolCall]:
        tools = self.registry.list()
        known = {t["name"] for t in tools}
        prompt_payload = {
            "user_message": message,
            "dataset_id": dataset_id,
            "dataset": self._planning_dataset_context(df),
            "rag_context": self.rag_context(citations),
            "available_tools": tools,
        }

        content = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a staff-level data analyst agent planner. "
                        "Select the smallest useful set of analytics tools. "
                        "Return strict JSON only with shape "
                        "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}. "
                        "Use rag_context for domain guidance when it is relevant. Use only tools "
                        "from available_tools. Do not invent columns; leave ambiguous tool "
                        "arguments empty so the executor can infer safe defaults."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, default=str)},
            ],
            temperature=0.0,
        )

        try:
            parsed = self._extract_json(content)
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
        except Exception as exc:
            raise LLMUnavailable(f"LLM planner returned invalid tool JSON: {exc}") from exc

        return calls[: settings.llm_max_tool_calls]

    def synthesize(
        self,
        message: str,
        *,
        dataset_id: str | None,
        dataset_context: dict[str, Any] | None,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        citations: list[dict],
    ) -> str:
        payload = {
            "user_message": message,
            "dataset_id": dataset_id,
            "dataset_context": dataset_context,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "tool_results": [tr.model_dump() for tr in tool_results],
            "rag_context": self.rag_context(citations),
        }

        return self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise data analyst. Explain what was done and summarize "
                        "the most important dataset-specific findings in plain language. Use "
                        "dataset_context for schema, missingness, distributions, examples, and "
                        "correlations; use tool_results for computed analytics. Do not expose "
                        "rag_context for relevant domain guidance. Do not expose hidden "
                        "chain-of-thought. If evidence is insufficient, say what input or tool "
                        "output is needed next. Ground your answer only in dataset_context, "
                        "tool_results, and rag_context."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=settings.llm_temperature,
        ).strip()
