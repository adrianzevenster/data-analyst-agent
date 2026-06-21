from __future__ import annotations

from typing import Any, Generator

import pandas as pd
from pydantic import ValidationError

from app.agent.llm import LATEST_TRAINED_MODEL_SENTINEL
from app.analytics.dataset_manager import DatasetManager
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.tooling import get_registry
from app.analytics.tool_validation import format_validation_error, validate_tool_args
from app.analytics.viz_specs import simple_bar_spec, multi_series_bar_spec, line_spec

from app.core.models import ToolCall, ToolResult

CHART_SPEC_TYPES = {"bar", "histogram", "line", "scatter"}


def _infer_numeric_cols(df: pd.DataFrame, max_cols: int = 10) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]


def _df_table_payload(df: pd.DataFrame, title: str) -> dict:
    import numpy as np
    d = df.head(200).copy()
    records = d.to_dict(orient="records")

    def _coerce(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    safe_records = [{k: _coerce(val) for k, val in row.items()} for row in records]
    return {"title": title, "columns": [str(c) for c in d.columns], "data": safe_records}


def _safe_json(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return {"type": "dataframe", "rows": int(obj.shape[0]), "cols": int(obj.shape[1])}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _flatten_dict(d: dict, prefix: str = "") -> list[dict]:
    import numpy as np
    rows = []

    for key, value in d.items():
        metric = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            rows.extend(_flatten_dict(value, metric))
        elif isinstance(value, list):
            rows.append({"metric": metric, "value": f"{len(value)} items"})
        elif isinstance(value, (np.integer,)):
            rows.append({"metric": metric, "value": int(value)})
        elif isinstance(value, (np.floating,)):
            rows.append({"metric": metric, "value": None if np.isnan(value) else float(value)})
        elif isinstance(value, (np.bool_,)):
            rows.append({"metric": metric, "value": bool(value)})
        else:
            rows.append({"metric": metric, "value": str(value)})

    return rows


class Executor:
    def __init__(self):
        self.dm = DatasetManager()
        self.model_manager = ModelManager()
        self.registry = get_registry()

    def run_stream(
        self, dataset_id: str, calls: list[ToolCall]
    ) -> Generator[tuple[ToolResult, list[dict], list[dict]], None, None]:
        """Yields (ToolResult, tables, charts) after each tool completes."""
        df = self.dm.load_df(dataset_id)
        tool_results_so_far: list[ToolResult] = []

        for call in calls:
            call_tables: list[dict] = []
            call_charts: list[dict] = []

            try:
                tool = self.registry.get(call.name)
                args = dict(call.arguments or {})

                # Tool-specific defaults
                if call.name == "multidim_pivot":
                    if not args.get("values"):
                        nums = _infer_numeric_cols(df, 1)
                        if not nums:
                            raise ValueError("No numeric columns found for pivot values.")
                        args["values"] = nums[0]
                    if not args.get("index"):
                        cats = [c for c in df.columns if df[c].dtype == "object"][:2]
                        if not cats:
                            cats = [str(df.columns[0])]
                        args["index"] = cats

                if call.name in ("anomaly_scan", "kmeans_clusters"):
                    if not args.get("numeric_cols"):
                        args["numeric_cols"] = _infer_numeric_cols(df, 8)

                if call.name in ("score_with_model", "explain_model") and args.get("model_id") == LATEST_TRAINED_MODEL_SENTINEL:
                    resolved_id = next(
                        (
                            tr.result.get("model_id")
                            for tr in reversed(tool_results_so_far)
                            if tr.name == "train_supervised_model"
                            and tr.ok
                            and isinstance(tr.result, dict)
                            and tr.result.get("model_id")
                        ),
                        None,
                    )
                    if not resolved_id:
                        raise ValueError(
                            "No model was trained earlier in this request to score with; "
                            "train a model first or name an existing model_id."
                        )
                    args["model_id"] = resolved_id

                try:
                    args = tool.validate_args(args)
                except ValidationError as e:
                    raise ValueError(f"Invalid arguments for {call.name}: {format_validation_error(e)}") from e

                validate_tool_args(df, call.name, args)

                # Execute tool
                extra_kwargs: dict[str, Any] = {}
                if call.name == "train_supervised_model":
                    extra_kwargs = {"model_manager": self.model_manager, "dataset_id": dataset_id}
                elif call.name in ("score_with_model", "explain_model", "evaluate_trained_model"):
                    extra_kwargs = {"model_manager": self.model_manager}
                elif call.name == "duckdb_query":
                    from app.analytics.sql import _safe_table_name
                    extra_tables: dict[str, pd.DataFrame] = {}
                    for meta in self.dm.list_datasets():
                        if meta.dataset_id != dataset_id:
                            try:
                                other_df = self.dm.load_df(meta.dataset_id)
                                tname = _safe_table_name(meta.filename)
                                if tname == "t":
                                    tname = f"t_{meta.dataset_id[:6]}"
                                extra_tables[tname] = other_df
                            except Exception:
                                pass
                    if extra_tables:
                        extra_kwargs["extra_tables"] = extra_tables

                result = tool.fn(df, **args, **extra_kwargs)
                tool_result = ToolResult(name=call.name, ok=True, result=_safe_json(result))
                tool_results_so_far.append(tool_result)

                # Attach UI payloads
                if isinstance(result, pd.DataFrame):
                    title = "Query Results" if call.name == "duckdb_query" else call.name
                    call_tables.append(_df_table_payload(result, title=title))

                    if result.shape[1] >= 2:
                        x = str(result.columns[0])
                        numeric_y_cols = [
                            str(c) for c in result.columns[1:] if pd.api.types.is_numeric_dtype(result[c])
                        ]

                        if numeric_y_cols:
                            if pd.api.types.is_datetime64_any_dtype(result[x]):
                                call_charts.append(
                                    line_spec(result, x=x, y=numeric_y_cols[0], title=f"{call.name}: {numeric_y_cols[0]} over {x}")
                                )
                            elif len(numeric_y_cols) > 1:
                                call_charts.append(
                                    multi_series_bar_spec(result, x=x, y_cols=numeric_y_cols, title=f"{call.name} by {x}")
                                )
                            else:
                                call_charts.append(
                                    simple_bar_spec(result, x=x, y=numeric_y_cols[0], title=f"{call.name}: {numeric_y_cols[0]} by {x}")
                                )

                elif isinstance(result, dict):
                    if result.get("type") in CHART_SPEC_TYPES and "data" in result:
                        call_charts.append(result)
                    else:
                        embedded_charts = result.get("charts")
                        if isinstance(embedded_charts, list):
                            call_charts.extend(c for c in embedded_charts if isinstance(c, dict))

                        list_table_keys = [
                            key for key, value in result.items()
                            if key != "charts" and isinstance(value, list) and value and isinstance(value[0], dict)
                        ]
                        for key in list_table_keys:
                            call_tables.append(_df_table_payload(pd.DataFrame(result[key]), title=f"{call.name}_{key}"))

                        remaining = {
                            k: v for k, v in result.items() if k not in list_table_keys and k != "charts"
                        }
                        flat_rows = _flatten_dict(remaining)
                        if flat_rows:
                            call_tables.append(
                                {
                                    "title": call.name,
                                    "columns": ["metric", "value"],
                                    "data": flat_rows,
                                }
                            )

                elif isinstance(result, list):
                    if result and isinstance(result[0], dict):
                        call_tables.append(_df_table_payload(pd.DataFrame(result), title=call.name))

            except Exception as e:
                tool_result = ToolResult(name=call.name, ok=False, error=str(e))
                tool_results_so_far.append(tool_result)

            yield tool_result, call_tables, call_charts

    def run(self, dataset_id: str, calls: list[ToolCall]) -> tuple[list[ToolResult], list[dict], list[dict]]:
        all_results: list[ToolResult] = []
        all_tables: list[dict] = []
        all_charts: list[dict] = []
        for result, tables, charts in self.run_stream(dataset_id, calls):
            all_results.append(result)
            all_tables.extend(tables)
            all_charts.extend(charts)
        return all_results, all_tables, all_charts
