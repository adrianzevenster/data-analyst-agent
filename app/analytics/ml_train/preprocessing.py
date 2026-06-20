from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

MAX_CATEGORICAL_CARDINALITY = 50
MAX_ORDINAL_CARDINALITY = 500
# If a column's unique-value fraction exceeds this, it looks like an identifier
# (e.g. customer_id, note text) and is dropped even within the ordinal range.
MAX_ORDINAL_UNIQUE_FRACTION = 0.5

# Mean word-count threshold for classifying a dropped string column as free text
# rather than a high-cardinality code/ID column.
_TEXT_WORD_COUNT_THRESHOLD = 3.0
_TEXT_DETECTION_SAMPLE = 200
TEXT_EMBEDDING_N_COMPONENTS = 64

# Module-level LocalEmbedder singleton — shared across all TextEmbeddingEncoder
# instances so the SentenceTransformer model is loaded once per process and is
# never serialised with joblib (only the computed lookup cache is).
_EMBEDDER: Any = None


def _get_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from app.rag.embedder import LocalEmbedder
        _EMBEDDER = LocalEmbedder()
    return _EMBEDDER


class TextEmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Encodes free-text columns into dense float vectors via LocalEmbedder.

    fit() builds a {text: embedding} lookup over unique training strings, then
    fits TruncatedSVD to reduce the raw 384-dim sentence embeddings to
    n_components dims (default 64).  This improves generalisation on small
    datasets and makes text columns cheap enough to include in auto model
    selection (6× fewer dimensions to process per CV fold).

    Only the SVD projection matrix and the compact lookup cache are serialised
    by joblib; the SentenceTransformer model lives in a module-level singleton.
    """

    def __init__(self, n_components: int = TEXT_EMBEDDING_N_COMPONENTS):
        self.n_components = n_components

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._columns: list[str] = [str(c) for c in df.columns]
        self._raw_dim: int = 0
        self._output_dim: int = 0
        self._cache: dict[str, np.ndarray] = {}
        self._svd: TruncatedSVD | None = None
        self._skipped: bool = False
        self._skip_reason: str = ""

        unique_texts: list[str] = list({
            text
            for col in self._columns
            for text in df[col].fillna("").astype(str).unique()
        })

        if unique_texts:
            try:
                embedder = _get_embedder()
                vecs = np.array(embedder.embed(unique_texts), dtype="float32")
                self._raw_dim = vecs.shape[1] if vecs.ndim == 2 else 0
                self._cache = {t: vecs[i] for i, t in enumerate(unique_texts)}

                # Cap n_components to satisfy TruncatedSVD constraints: must be
                # strictly less than both n_samples and n_features.
                n_comp = min(self.n_components, len(unique_texts) - 1, self._raw_dim - 1)
                if n_comp >= 1:
                    self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    self._svd.fit(vecs)
                    self._output_dim = n_comp
                else:
                    self._output_dim = self._raw_dim
            except Exception as exc:
                # Embedder unavailable (model not cached, no network access).
                # Treat text columns as dropped rather than failing the pipeline.
                self._columns = []
                self._skipped = True
                self._skip_reason = str(exc)

        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if self._raw_dim == 0 or not self._columns:
            return np.zeros((len(df), 0), dtype="float32")

        oov = list({
            t
            for col in self._columns
            for t in df[col].fillna("").astype(str).unique()
            if t not in self._cache
        })
        if oov:
            embedder = _get_embedder()
            new_vecs = np.array(embedder.embed(oov), dtype="float32")
            self._cache.update(zip(oov, new_vecs))

        zero = np.zeros(self._raw_dim, dtype="float32")
        parts: list[np.ndarray] = []
        for col in self._columns:
            texts = df[col].fillna("").astype(str).tolist()
            col_embs = np.stack([self._cache.get(t, zero) for t in texts])
            if self._svd is not None:
                col_embs = self._svd.transform(col_embs).astype("float32")
            parts.append(col_embs)

        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        names = [
            f"{col}__emb_{i}"
            for col in self._columns
            for i in range(self._output_dim)
        ]
        return np.array(names, dtype=object)


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts numeric calendar features from datetime-typed columns.

    Emits year, month, day, dayofweek, is_weekend and (when non-midnight
    times are present) hour per input column, so pipeline feature-importance
    methods can name and rank them correctly.
    """

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self._columns: list[str] = [str(c) for c in df.columns]
        self._has_hour: dict[str, bool] = {}
        for col in self._columns:
            s = pd.to_datetime(df[col], errors="coerce")
            self._has_hour[col] = bool((s.dt.hour != 0).any())
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        parts: list[pd.DataFrame] = []
        for col in self._columns:
            s = pd.to_datetime(df[col], errors="coerce")
            col_data: dict[str, pd.Series] = {
                f"{col}__year": s.dt.year,
                f"{col}__month": s.dt.month,
                f"{col}__day": s.dt.day,
                f"{col}__dayofweek": s.dt.dayofweek,
                f"{col}__is_weekend": s.dt.dayofweek.isin([5, 6]).astype(int),
            }
            if self._has_hour.get(col, False):
                col_data[f"{col}__hour"] = s.dt.hour
            parts.append(pd.DataFrame(col_data, index=df.index))
        if not parts:
            return np.zeros((len(df), 0), dtype="float32")
        return pd.concat(parts, axis=1).fillna(0).to_numpy(dtype="float32")

    def get_feature_names_out(self, input_features=None):
        names: list[str] = []
        for col in self._columns:
            names += [
                f"{col}__year",
                f"{col}__month",
                f"{col}__day",
                f"{col}__dayofweek",
                f"{col}__is_weekend",
            ]
            if self._has_hour.get(col, False):
                names.append(f"{col}__hour")
        return np.array(names, dtype=object)


def _is_text_col(series: pd.Series, sample: int = _TEXT_DETECTION_SAMPLE) -> bool:
    """True when a string column looks like free text (mean word count > threshold).

    IDs and codes have no spaces → 1 word. Location strings like 'New York'
    → 2 words. Descriptions/notes → 4+ words. The threshold of 3 draws a
    line that passes IDs and short categoricals but catches narrative text.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return False
    if len(s) > sample:
        s = s.sample(n=sample, random_state=42)
    return float(s.str.split().str.len().mean()) > _TEXT_WORD_COUNT_THRESHOLD


def split_feature_types(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Returns (numeric, ohe_categorical, ordinal_categorical, datetime, text, dropped).

    Datetime columns → DatetimeFeatureExtractor (calendar features).
    Columns with ≤50 unique values → OHE.
    Columns with 51–500 unique values AND unique fraction ≤50% of rows →
        OrdinalEncoder (rescues genuine high-cardinality features like zip codes).
    High-cardinality string columns where mean word count >3 → TextEmbeddingEncoder.
    Otherwise → dropped (identifiers, extreme cardinality codes).
    """
    datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    remaining = [c for c in feature_cols if c not in set(datetime_cols)]

    numeric_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in remaining if c not in numeric_cols]

    n_rows = max(len(df), 1)
    ohe_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) <= MAX_CATEGORICAL_CARDINALITY]
    ordinal_cols = [
        c for c in categorical_cols
        if MAX_CATEGORICAL_CARDINALITY < df[c].nunique(dropna=True) <= MAX_ORDINAL_CARDINALITY
        and df[c].nunique(dropna=True) / n_rows <= MAX_ORDINAL_UNIQUE_FRACTION
    ]
    initially_dropped = [c for c in categorical_cols if c not in ohe_cols and c not in ordinal_cols]

    # Rescue free-text columns from the dropped set.
    text_cols = [c for c in initially_dropped if _is_text_col(df[c])]
    dropped = [c for c in initially_dropped if c not in set(text_cols)]

    return numeric_cols, ohe_cols, ordinal_cols, datetime_cols, text_cols, dropped


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: list[str] | None = None,
    datetime_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
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

    if datetime_cols:
        transformers.append(("datetime", DatetimeFeatureExtractor(), datetime_cols))

    if text_cols:
        transformers.append(("text", TextEmbeddingEncoder(), text_cols))

    return ColumnTransformer(transformers)
