"""Auto-EDA cache: run profile_dataset + auto_insights on upload and store results."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)
_CACHE_DIR = "eda_cache"


def _cache_path(dataset_id: str) -> Path:
    d = settings.data_path / _CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset_id}.json"


def get_cached(dataset_id: str) -> dict | None:
    p = _cache_path(dataset_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def run_and_cache(dataset_id: str, df: pd.DataFrame) -> None:
    """Run profile + insights synchronously and write result to cache.

    Called as a FastAPI BackgroundTask after upload so the upload response
    is not blocked.  Overwrites any previous cache for this dataset_id.
    """
    from app.analytics.profiling import profile_dataset
    from app.analytics.insights import auto_insights

    try:
        profile = profile_dataset(df, sample=min(len(df), 5000))
        insights = auto_insights(df, top_n=8)
        result = {
            "dataset_id": dataset_id,
            "ready": True,
            "profile": profile,
            "insights": insights,
        }
        _cache_path(dataset_id).write_text(json.dumps(result, default=str))
        logger.info("EDA cached for dataset %s", dataset_id)
    except Exception as exc:
        logger.warning("EDA cache failed for %s: %s", dataset_id, exc)
        # Write a minimal stub so the frontend knows EDA attempted but failed.
        try:
            _cache_path(dataset_id).write_text(
                json.dumps({"dataset_id": dataset_id, "ready": False, "error": str(exc)})
            )
        except Exception:
            pass
