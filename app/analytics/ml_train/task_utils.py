from __future__ import annotations

import re

import pandas as pd

_ROW_NUM_RE = re.compile(
    r'^(row[_\s]?(num(ber)?|id|index|no)|rownum|rowid|rowindex|'
    r'record[_\s]?(num(ber)?|id|no)|line[_\s]?(num(ber)?|no)|'
    r'seq(uence)?[_\s]?num(ber)?)$',
    re.IGNORECASE,
)


def _looks_like_id_col(df: pd.DataFrame, col: str) -> bool:
    """True when a numeric column is an identifier or sequential row number."""
    name = col.strip().lower().replace(" ", "_").replace("-", "_")
    if _ROW_NUM_RE.match(name):
        return True
    if pd.api.types.is_integer_dtype(df[col]):
        s = df[col].dropna()
        n = len(s)
        if n > 1 and s.nunique() == n:
            vals = s.sort_values().to_numpy()
            if int(vals[-1]) - int(vals[0]) == n - 1:
                return True
    return False


def _should_log_transform(y: pd.Series) -> bool:
    """True when a regression target is non-negative and skewness > 1.5."""
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty or float(numeric.min()) < 0:
        return False
    try:
        return float(numeric.skew()) > 1.5
    except Exception:
        return False


def _infer_task_type(y: pd.Series, task_hint: str) -> str:
    if task_hint != "auto":
        return task_hint
    numeric = pd.to_numeric(y, errors="coerce").notna().mean() >= 0.90
    if numeric:
        unique = pd.to_numeric(y, errors="coerce").nunique(dropna=True)
        return "classification" if unique <= 20 else "regression"
    return "classification"


def _readout(task_type: str, model_type: str, n_train: int, n_test: int, model_id: str, evaluation: dict) -> str:
    if task_type == "classification":
        accuracy = evaluation.get("accuracy")
        metric = f"accuracy={accuracy:.4f}" if accuracy is not None else "metrics computed"
    else:
        wmape = evaluation.get("wmape")
        metric = f"WMAPE={wmape:.4f}" if wmape is not None else "metrics computed"
    return (
        f"Trained {model_type} ({task_type}) on {n_train} rows, evaluated on {n_test} held-out rows "
        f"({metric}). Model persisted as {model_id}."
    )
