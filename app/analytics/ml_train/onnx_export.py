"""Optional ONNX export for standard sklearn pipelines.

Conversion is attempted only when the pipeline contains no custom
transformers (TextEmbeddingEncoder, DatetimeFeatureExtractor,
InteractionFeatureTransformer).  If conversion or validation fails for any
reason the function returns None and training continues unaffected.

Requires:  skl2onnx  onnxruntime  (both optional — imported lazily).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Custom transformer class names that skl2onnx cannot convert.
_UNCONVERTIBLE = frozenset({
    "TextEmbeddingEncoder",
    "DatetimeFeatureExtractor",
    "InteractionFeatureTransformer",
})


def _has_custom_transformer(pipeline: Any) -> bool:
    """Walk the pipeline tree and return True if any custom transformer is found."""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    def _walk(estimator: Any) -> bool:
        name = type(estimator).__name__
        if name in _UNCONVERTIBLE:
            return True
        if isinstance(estimator, Pipeline):
            return any(_walk(step) for _, step in estimator.steps)
        if isinstance(estimator, ColumnTransformer):
            for tname, t, _ in estimator.transformers_:
                if tname == "remainder":
                    continue
                if _walk(t):
                    return True
        return False

    return _walk(pipeline)


def try_export_onnx(
    pipeline: Any,
    X_sample: pd.DataFrame,
    model_id: str,
    model_dir: Path,
) -> str | None:
    """Convert pipeline to ONNX, validate with onnxruntime, return path or None.

    Skips silently when:
    - skl2onnx / onnxruntime are not installed
    - The pipeline contains custom (non-ONNX-convertible) transformers
    - Conversion or validation raises any exception
    """
    try:
        import onnxruntime as rt
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import (
            FloatTensorType,
            Int64TensorType,
            StringTensorType,
        )
    except ImportError:
        return None

    if _has_custom_transformer(pipeline):
        return None

    try:
        # Build initial_types from X_sample column dtypes.
        initial_types: list[tuple[str, Any]] = []
        for col in X_sample.columns:
            dtype = X_sample[col].dtype
            if pd.api.types.is_float_dtype(dtype):
                t = FloatTensorType([None, 1])
            elif pd.api.types.is_integer_dtype(dtype):
                t = Int64TensorType([None, 1])
            else:
                t = StringTensorType([None, 1])
            initial_types.append((str(col), t))

        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_types,
            options={"zipmap": False},  # return plain arrays for classifiers
        )

        # Validate: run a forward pass with onnxruntime.
        sess = rt.InferenceSession(onnx_model.SerializeToString())
        feed: dict[str, np.ndarray] = {}
        for col in X_sample.columns:
            dtype = X_sample[col].dtype
            arr = X_sample[[col]].to_numpy()
            if pd.api.types.is_float_dtype(dtype):
                feed[col] = arr.astype(np.float32)
            elif pd.api.types.is_integer_dtype(dtype):
                feed[col] = arr.astype(np.int64)
            else:
                feed[col] = arr.astype(str)
        sess.run(None, feed)

        # Persist.
        onnx_path = model_dir / f"{model_id}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info("ONNX export succeeded for model %s → %s", model_id, onnx_path)
        return str(onnx_path)

    except Exception as exc:
        logger.debug("ONNX export skipped for model %s: %s", model_id, exc)
        return None
