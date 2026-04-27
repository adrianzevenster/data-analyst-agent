from __future__ import annotations

import pandas as pd

from app.analytics.dataset_manager import DatasetManager
from app.analytics.tooling import get_registry
from app.analytics.viz_specs import simple_bar_spec

from app.core.models import ToolCall, ToolResult


def _infer_numeric_cols(df: pd.DataFrame, max_cols: int = 10) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]


def _df_table_payload(df: pd.DataFrame, title: str) -> dict:
    d = df.head(200).copy()
    return {"title": title, "columns": [str(c) for c in d.columns], "data": d.to_dict(orient="records")}


def _safe_json(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return {"type": "dataframe", "rows": int(obj.shape[0]), "cols": int(obj.shape[1])}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class Executor:
    def __init__(self):
        self.dm = DatasetManager()
        self.registry = get_registry()

    def run(self, dataset_id: str, calls: list[ToolCall]) -> tuple[list[ToolResult], list[dict], list[dict]]:
        df = self.dm.load_df(dataset_id)
        tool_results: list[ToolResult] = []
        tables: list[dict] = []
        charts: list[dict] = []

        for call in calls:
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

                # Execute tool
                result = tool.fn(df, **args)
                tool_results.append(ToolResult(name=call.name, ok=True, result=_safe_json(result)))

                # Attach UI payloads
                if isinstance(result, pd.DataFrame):
                    tables.append(_df_table_payload(result, title=call.name))

                    # Optional chart for tabular outputs
                    if result.shape[1] >= 2:
                        x = str(result.columns[0])
                        y = next((str(c) for c in result.columns[1:] if pd.api.types.is_numeric_dtype(result[c])), None)
                        if y:
                            charts.append(simple_bar_spec(result, x=x, y=y, title=f"{call.name}: {y} by {x}"))

                elif isinstance(result, dict):
                    if "columns" in result and isinstance(result["columns"], list):
                        cols_df = pd.DataFrame(result["columns"])
                        tables.append(_df_table_payload(cols_df, title=f"{call.name}_columns"))

                elif isinstance(result, list):
                    # NEW: handle list[dict] results (e.g., skewed_features)
                    if result and isinstance(result[0], dict):
                        tables.append(_df_table_payload(pd.DataFrame(result), title=call.name))

            except Exception as e:
                tool_results.append(ToolResult(name=call.name, ok=False, error=str(e)))

        return tool_results, tables, charts
