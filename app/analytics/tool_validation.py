from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import ValidationError


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
