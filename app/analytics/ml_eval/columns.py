from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PredictionColumns:
    actual_col: str | None
    prediction_col: str | None
    probability_col: str | None
    id_col: str | None


ACTUAL_CANDIDATES = [
    "actual",
    "target",
    "label",
    "y_true",
    "ground_truth",
    "churn",
]

PREDICTION_CANDIDATES = [
    "prediction",
    "predicted",
    "y_pred",
    "churn prediction",
    "predicted classification",
    "classification",
]

PROBABILITY_CANDIDATES = [
    "probability",
    "predicted probability",
    "prediction probability",
    "churn probability",
    "score",
    "confidence",
]

ID_CANDIDATES = [
    "id",
    "customer_id",
    "customer id",
    "account number",
    "ledger acc id",
]


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {_normalise(c): c for c in df.columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    for col in df.columns:
        norm = _normalise(col)
        if any(candidate in norm for candidate in candidates):
            return str(col)

    return None


def infer_prediction_columns(
        df: pd.DataFrame,
        actual_col: str | None = None,
        prediction_col: str | None = None,
        probability_col: str | None = None,
        id_col: str | None = None,
) -> PredictionColumns:
    inferred_probability_col = probability_col or find_column(df, PROBABILITY_CANDIDATES)

    inferred_actual_col = actual_col or find_column(df, ACTUAL_CANDIDATES)
    inferred_prediction_col = prediction_col or find_column(df, PREDICTION_CANDIDATES)

    if (
            inferred_probability_col
            and inferred_actual_col
            and not prediction_col
            and _normalise(inferred_actual_col) in {"churn", "renewal"}
    ):
        inferred_prediction_col = inferred_actual_col
        inferred_actual_col = None

    return PredictionColumns(
        actual_col=inferred_actual_col,
        prediction_col=inferred_prediction_col,
        probability_col=inferred_probability_col,
        id_col=id_col or find_column(df, ID_CANDIDATES),
    )