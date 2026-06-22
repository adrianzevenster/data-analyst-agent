"""Thread-safe in-memory job registry for background model training."""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

_MAX_JOBS = 50
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg-train")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def submit_job(fn: Callable, *args: Any, **kwargs: Any) -> str:
    """Submit a callable to the background pool. Returns a job_id immediately."""
    job_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()

    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "created_at": created,
            "completed_at": None,
        }
        _evict_oldest()

    def _run() -> None:
        try:
            result = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
                _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    _pool.submit(_run)
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        entry = _jobs.get(job_id)
        return dict(entry) if entry else None


def list_jobs(limit: int = 20) -> list[dict]:
    with _lock:
        items = [{"job_id": k, **v} for k, v in _jobs.items()]
    items.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return items[:limit]


def _evict_oldest() -> None:
    """Keep the registry bounded. Called inside _lock."""
    if len(_jobs) < _MAX_JOBS:
        return
    oldest = sorted(_jobs.keys(), key=lambda k: _jobs[k].get("created_at", ""))
    for k in oldest[: len(_jobs) - _MAX_JOBS + 1]:
        del _jobs[k]
