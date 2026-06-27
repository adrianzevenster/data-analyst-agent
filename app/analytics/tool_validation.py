from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_MODEL_SENTINEL = "<latest_trained_model_id>"


def format_validation_error(exc: ValidationError) -> str:
    details = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "Invalid value")
        details.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(details)


def validate_columns(df: pd.DataFrame, columns: list[str] | None, arg_name: str) -> None:
    if not columns:
        return
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{arg_name} contains unknown columns: {missing}")


def validate_tool_args(df: pd.DataFrame, call_name: str, args: dict[str, Any]) -> None:
    """Schema-grounds tool arguments against the actual dataset columns.

    Shared by the executor (final guard before running a tool) and the LLM
    planner (so a hallucinated column can be caught and repaired before
    ever reaching the executor).
    """
    if call_name == "multidim_pivot":
        validate_columns(df, args.get("index"), "index")
        validate_columns(df, args.get("columns"), "columns")
        values = args.get("values")
        if values and values not in df.columns:
            raise ValueError(f"values column not in dataset: {values}")

    if call_name in {"anomaly_scan", "kmeans_clusters"}:
        validate_columns(df, args.get("numeric_cols"), "numeric_cols")

    if call_name == "evaluate_ml_predictions":
        for key in ("actual_col", "prediction_col", "probability_col", "id_col"):
            value = args.get(key)
            if value and value not in df.columns:
                raise ValueError(f"{key} not in dataset: {value}")
        validate_columns(df, args.get("slice_cols"), "slice_cols")

    if call_name == "train_supervised_model":
        target_col = args.get("target_col")
        if target_col and target_col not in df.columns:
            raise ValueError(f"target_col not in dataset: {target_col}")
        validate_columns(df, args.get("feature_cols"), "feature_cols")

    if call_name in {"simple_bar_spec", "line_spec", "scatter_spec"}:
        for key in ("x", "y"):
            value = args.get(key)
            if value and value not in df.columns:
                raise ValueError(f"{key} not in dataset: {value}")

    if call_name == "histogram_spec":
        column = args.get("column")
        if column and column not in df.columns:
            raise ValueError(f"column not in dataset: {column}")

    if call_name == "correlation_analysis":
        validate_columns(df, args.get("numeric_cols"), "numeric_cols")
        validate_columns(df, args.get("categorical_cols"), "categorical_cols")

    if call_name == "trend_analysis":
        for key in ("date_col", "value_col"):
            value = args.get(key)
            if value and value not in df.columns:
                raise ValueError(f"{key} not in dataset: {value}")

    if call_name == "overrepresented_categories":
        col = args.get("col")
        if col and col not in df.columns:
            raise ValueError(f"col not in dataset: {col}")

    if call_name == "skewed_features":
        validate_columns(df, args.get("cols"), "cols")

    if call_name == "missingness_matrix":
        validate_columns(df, args.get("cols"), "cols")

    if call_name in {"explain_model", "score_with_model", "evaluate_trained_model",
                     "shap_explain_prediction", "forecast_with_model"}:
        model_id = str(args.get("model_id") or "")
        if model_id and model_id != _MODEL_SENTINEL and not _UUID_RE.match(model_id):
            raise ValueError(
                f"model_id '{model_id}' is not a valid UUID. "
                "Use an ID from known_trained_model_ids, or omit this call if no model has been trained yet."
            )

    if call_name == "duckdb_query":
        query = args.get("query", "")
        if query:
            import duckdb
            try:
                con = duckdb.connect(database=":memory:", config={"enable_external_access": False})
                con.register("t", df.head(0))
                con.execute(f"EXPLAIN {query}")
                con.close()
            except Exception as exc:
                actual_cols = ", ".join(str(c) for c in df.columns)
                raise ValueError(
                    f"SQL references unknown columns or has a syntax error. "
                    f"Table 't' has columns: [{actual_cols}]. Error: {exc}"
                ) from exc
