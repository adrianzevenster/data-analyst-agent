from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

MAX_CATEGORICAL_CARDINALITY = 50
MAX_ORDINAL_CARDINALITY = 500
# If a column's unique-value fraction exceeds this, it looks like an identifier
# (e.g. customer_id, note text) and is dropped even within the ordinal range.
MAX_ORDINAL_UNIQUE_FRACTION = 0.5


def split_feature_types(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (numeric, ohe_categorical, ordinal_categorical, dropped).

    Columns with ≤50 unique values → OHE.
    Columns with 51–500 unique values AND unique fraction ≤50% of rows →
        OrdinalEncoder (rescues genuine high-cardinality features like zip codes).
    Otherwise → dropped (free text, identifiers, or extreme cardinality).
    """
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    n_rows = max(len(df), 1)
    ohe_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY]
    ordinal_cols = [
        c for c in categorical_cols
        if MAX_CATEGORICAL_CARDINALITY < df[c].nunique(dropna=True) <= MAX_ORDINAL_CARDINALITY
        and df[c].nunique(dropna=True) / n_rows <= MAX_ORDINAL_UNIQUE_FRACTION
    ]
    dropped = [c for c in categorical_cols if c not in ohe_cols and c not in ordinal_cols]

    return numeric_cols, ohe_cols, ordinal_cols, dropped


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: list[str] | None = None,
) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if ordinal_cols:
        ordinal_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

    return ColumnTransformer(transformers)
