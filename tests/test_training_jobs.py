"""Tests for the async training job store (submit, poll, error handling)."""
from __future__ import annotations

import time


def test_submit_and_get_job():
    from app.api.training_jobs import submit_job, get_job

    results = []
    job_id = submit_job(lambda: results.append(42) or {"done": True})
    job = get_job(job_id)
    assert job is not None
    assert job["status"] in ("running", "done")
    for _ in range(20):
        j = get_job(job_id)
        if j and j["status"] == "done":
            break
        time.sleep(0.1)
    final = get_job(job_id)
    assert final is not None
    assert final["status"] == "done"
    assert final["result"] == {"done": True}


def test_job_captures_errors():
    from app.api.training_jobs import submit_job, get_job

    job_id = submit_job(lambda: 1 / 0)
    for _ in range(20):
        j = get_job(job_id)
        if j and j["status"] == "error":
            break
        time.sleep(0.1)
    final = get_job(job_id)
    assert final is not None
    assert final["status"] == "error"
    assert "division by zero" in str(final.get("error", ""))


def test_list_jobs():
    from app.api.training_jobs import submit_job, list_jobs

    submit_job(lambda: {"x": 1})
    jobs = list_jobs(limit=50)
    assert len(jobs) >= 1
    assert all("job_id" in j for j in jobs)
    assert all("status" in j for j in jobs)
