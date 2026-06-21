from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    TargetEncoder,
)

MAX_CATEGORICAL_CARDINALITY = 50
MAX_ORDINAL_CARDINALITY = 500
MAX_ORDINAL_UNIQUE_FRACTION = 0.5

_TEXT_WORD_COUNT_THRESHOLD = 3.0
_TEXT_DETECTION_SAMPLE = 200
TEXT_EMBEDDING_N_COMPONENTS = 64

# Default lags and rolling windows for temporal feature engineering.
LAG_DEFAULTS: list[int] = [1, 7]
ROLLING_DEFAULTS: list[int] = [7]

_EMBEDDER: Any = None


def _get_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from app.rag.embedder import LocalEmbedder
        _EMBEDDER = LocalEmbedder()
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Lag / rolling feature engineering
# ---------------------------------------------------------------------------

def engineer_lag_features(
    df: pd.DataFrame,
    sort_col: str,
    lag_cols: list[str],
    lags: list[int] = LAG_DEFAULTS,
    windows: list[int] = ROLLING_DEFAULTS,
) -> tuple[pd.DataFrame, list[str]]:
    """Sort df by sort_col and create lag/rolling-mean features for lag_cols.

    Returns (new_df, new_col_names).  Rows where any strict lag feature is NaN
    (the first max(lags) rows after sorting) are dropped so every training row
    has a complete feature vector.
    """
    df = df.sort_values(sort_col).reset_index(drop=True)
    new_cols: list[str] = []

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            name = f"{col}__lag_{lag}"
            df[name] = df[col].shift(lag)
            new_cols.append(name)
        for w in windows:
            name = f"{col}__roll_mean_{w}"
            # shift(1) so the current row value isn't included in its own window
            df[name] = df[col].shift(1).rolling(w, min_periods=1).mean()
            new_cols.append(name)

    # Drop rows where strict lag features are NaN (insufficient history).
    lag_only = [c for c in new_cols if "__lag_" in c]
    if lag_only:
        df = df.dropna(subset=lag_only).reset_index(drop=True)

    return df, new_cols


# ---------------------------------------------------------------------------
# Text embedding encoder
# ---------------------------------------------------------------------------

class TextEmbeddingEncoder(BaseEstimator, TransformerMixin):
    """Encodes free-text columns into dense float vectors via LocalEmbedder.

    fit() builds a {text: embedding} lookup over unique training strings, then
    fits TruncatedSVD to reduce the raw 384-dim sentence embeddings to
    n_components dims (default 64).
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

                n_comp = min(self.n_components, len(unique_texts) - 1, self._raw_dim - 1)
                if n_comp >= 1:
                    self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    self._svd.fit(vecs)
                    self._output_dim = n_comp
                else:
                    self._output_dim = self._raw_dim
            except Exception as exc:
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


# ---------------------------------------------------------------------------
# Datetime feature extractor
# ---------------------------------------------------------------------------

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts numeric calendar features from datetime-typed columns."""

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


# ---------------------------------------------------------------------------
# Interaction feature transformer
# ---------------------------------------------------------------------------

class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Creates degree-2 pairwise interaction features for top-K numeric inputs.

    Selects up to max_input_features columns by variance, creates interaction-
    only PolynomialFeatures (no squared terms, no bias), drops zero-variance
    results, and caps output at max_output_features by keeping the highest-
    variance interactions.  The original features are NOT included in the
    output — they are already emitted by the numeric pipeline in the parent
    ColumnTransformer.
    """

    def __init__(self, max_input_features: int = 15, max_output_features: int = 50):
        self.max_input_features = max_input_features
        self.max_output_features = max_output_features

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        n_cols = arr.shape[1]

        if n_cols < 2:
            self._sel_idx: list[int] = list(range(n_cols))
            self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self._poly.fit(arr[:, self._sel_idx] if self._sel_idx else arr[:, :0])
            self._vt = VarianceThreshold(threshold=0.0)
            self._vt.fit(np.zeros((arr.shape[0], 0)))
            self._keep_idx: list[int] = []
            self._out_names: np.ndarray = np.array([], dtype=object)
            return self

        variances = np.nanvar(arr, axis=0)
        k = min(self.max_input_features, n_cols)
        self._sel_idx = np.argsort(variances)[::-1][:k].tolist()

        X_sel = arr[:, self._sel_idx]
        self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = self._poly.fit_transform(X_sel)

        # Drop the leading original-feature columns; keep only pairwise products.
        n_orig = len(self._sel_idx)
        X_interact = X_poly[:, n_orig:]

        if X_interact.shape[1] == 0:
            self._vt = VarianceThreshold(threshold=0.0)
            self._vt.fit(X_interact)
            self._keep_idx = []
            self._out_names = np.array([], dtype=object)
            return self

        self._vt = VarianceThreshold(threshold=0.0)
        X_filtered = self._vt.fit_transform(X_interact)

        n_out = X_filtered.shape[1]
        if n_out > self.max_output_features:
            out_vars = np.nanvar(X_filtered, axis=0)
            self._keep_idx = np.argsort(out_vars)[::-1][:self.max_output_features].tolist()
        else:
            self._keep_idx = list(range(n_out))

        raw_names = self._poly.get_feature_names_out([f"n{i}" for i in range(len(self._sel_idx))])
        interaction_names = raw_names[n_orig:]
        vt_mask = self._vt.get_support()
        filtered_names = interaction_names[vt_mask]
        self._out_names = filtered_names[self._keep_idx]

        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if not self._keep_idx:
            return np.zeros((arr.shape[0], 0), dtype="float32")

        X_sel = arr[:, self._sel_idx]
        X_poly = self._poly.transform(X_sel)
        n_orig = len(self._sel_idx)
        X_interact = X_poly[:, n_orig:]
        X_filtered = self._vt.transform(X_interact)
        return X_filtered[:, self._keep_idx].astype("float32")

    def get_feature_names_out(self, input_features=None):
        if not self._keep_idx:
            return np.array([], dtype=object)

        # Use real column names when the pipeline passes them through.
        if input_features is not None and len(input_features) > max(self._sel_idx, default=-1):
            sel_names = [str(input_features[i]) for i in self._sel_idx]
            raw_names = self._poly.get_feature_names_out(sel_names)
            n_orig = len(self._sel_idx)
            interaction_names = raw_names[n_orig:]
            vt_mask = self._vt.get_support()
            filtered_names = interaction_names[vt_mask]
            return np.array(filtered_names[self._keep_idx], dtype=object)

        return np.array(self._out_names, dtype=object)


# ---------------------------------------------------------------------------
# Column type detection helpers
# ---------------------------------------------------------------------------

def _is_text_col(series: pd.Series, sample: int = _TEXT_DETECTION_SAMPLE) -> bool:
    """True when a string column looks like free text (mean word count > threshold)."""
    s = series.dropna().astype(str)
    if s.empty:
        return False
    if len(s) > sample:
        s = s.sample(n=sample, random_state=42)
    return float(s.str.split().str.len().mean()) > _TEXT_WORD_COUNT_THRESHOLD


def split_feature_types(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Returns (numeric, ohe_categorical, ordinal_categorical, datetime, text, dropped)."""
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

    text_cols = [c for c in initially_dropped if _is_text_col(df[c])]
    dropped = [c for c in initially_dropped if c not in set(text_cols)]

    return numeric_cols, ohe_cols, ordinal_cols, datetime_cols, text_cols, dropped


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: list[str] | None = None,
    datetime_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    add_interactions: bool = False,
) -> ColumnTransformer:
    """Build a ColumnTransformer for the given feature sets.

    When add_interactions=True and numeric_cols has ≥2 columns, a second
    transformer for degree-2 pairwise interaction features is added alongside
    the numeric scaler.  TargetEncoder is used for ordinal_cols (high-
    cardinality categoricals with 51-500 unique values) instead of the
    arbitrary-rank OrdinalEncoder, which encodes the actual target-conditional
    signal and prevents leakage via internal cross-fitting.
    """
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if add_interactions and len(numeric_cols) >= 2:
            interaction_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("interact", InteractionFeatureTransformer()),
            ])
            transformers.append(("interactions", interaction_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if ordinal_cols:
        # TargetEncoder encodes each category as the target conditional mean,
        # using Bayesian shrinkage for rare categories and cv=5 cross-fitting
        # to prevent target leakage.  This is strictly better than OrdinalEncoder
        # for prediction tasks because it encodes the actual predictive signal.
        ordinal_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder(
                    target_type="auto",
                    smooth="auto",
                    cv=5,
                    random_state=42,
                )),
            ]
        )
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

    if datetime_cols:
        transformers.append(("datetime", DatetimeFeatureExtractor(), datetime_cols))

    if text_cols:
        transformers.append(("text", TextEmbeddingEncoder(), text_cols))

    return ColumnTransformer(transformers)
