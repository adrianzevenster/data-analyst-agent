import json
import threading
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.agent.judge_metrics import judge_metrics
from app.agent.latency_metrics import latency_metrics
from app.agent.llm_metrics import metrics
from app.analytics.quality_eval import QualityEvalPipeline
from app.core.config import settings
from app.core.models import (
    EvalRunHistoryEntry,
    EvalRunHistoryResponse,
    JudgeHistoryEntry,
    JudgeHistoryResponse,
    JudgeStatsResponse,
    LatencyStatsResponse,
    LLMStatsResponse,
    PlannerFallbackResponse,
    QualityTrendDay,
    QualityTrendResponse,
    RagEvalResponse,
    RepairStatsResponse,
)

_eval_pipeline = QualityEvalPipeline()

# ── eval job state ─────────────────────────────────────────────────────────────
_eval_jobs: dict[str, dict] = {}
_eval_jobs_lock = threading.Lock()

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/embed")
def embed_health():
    """Debug: show where the embedding model resolves to in this container."""
    import os
    from app.rag.embedder import _resolve_local_model_path, _DEFAULT_MODEL
    model = os.environ.get("EMBED_MODEL") or _DEFAULT_MODEL
    resolved = _resolve_local_model_path(model)
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    hf_offline = os.environ.get("HF_HUB_OFFLINE", "0")
    return {
        "model": model,
        "hf_home": hf_home,
        "hf_hub_offline": hf_offline,
        "resolved_path": resolved,
        "ready": resolved is not None or hf_offline == "1",
    }


@router.get("/health/llm", response_model=LLMStatsResponse)
def llm_health():
    return metrics.snapshot()


@router.get("/health/llm-judge", response_model=JudgeStatsResponse)
def llm_judge_health():
    return judge_metrics.snapshot()


@router.get("/health/llm-judge/history", response_model=JudgeHistoryResponse)
def llm_judge_history(limit: int = 500):
    entries = [JudgeHistoryEntry(**entry) for entry in judge_metrics.history(limit=min(limit, 2000))]
    return JudgeHistoryResponse(entries=entries, total=len(entries))


@router.get("/health/llm-repair", response_model=RepairStatsResponse)
def llm_repair_health():
    return metrics.repair_snapshot()


@router.get("/health/planner/fallback-rate", response_model=PlannerFallbackResponse)
def planner_fallback_health():
    return metrics.fallback_snapshot()


@router.get("/health/latency", response_model=LatencyStatsResponse)
def latency_health():
    return latency_metrics.snapshot()


@router.get("/health/rag-eval", response_model=RagEvalResponse)
def rag_eval_health():
    # This is a static report written by `pytest -m rag_eval`, not computed
    # live - recall/precision need a labeled query set and a real embedding
    # run, which isn't something to redo on every page load.
    report_path = settings.eval_path / "rag_retrieval_eval.json"
    if not report_path.exists():
        return RagEvalResponse(available=False)

    report = json.loads(report_path.read_text())
    return RagEvalResponse(
        available=True,
        n_queries=report["n_queries"],
        aggregate=report["aggregate"],
        min_recall_at_5=report.get("min_recall_at_5"),
    )


@router.get("/health/quality-trend", response_model=QualityTrendResponse)
def quality_trend(days: int = Query(default=30, ge=1, le=365)):
    """Daily aggregate groundedness scores from judge_log (request-time + eval runs)."""
    raw = _eval_pipeline.quality_trend(days=days)
    return QualityTrendResponse(
        days=days,
        data=[QualityTrendDay(**row) for row in raw],
    )


def _do_eval_run(run_id: str, n: int, max_age_days: int) -> None:
    """Background task: run eval pipeline, persist result into _eval_jobs."""
    from app.agent.llm import LLMReasoner
    from app.agent.executor import Executor

    try:
        reasoner = LLMReasoner()
        dm = Executor().dm
        result = _eval_pipeline.run(reasoner=reasoner, dm=dm, n=n, max_age_days=max_age_days)
        with _eval_jobs_lock:
            _eval_jobs[run_id] = {"status": "done", **result}
    except Exception as exc:
        with _eval_jobs_lock:
            _eval_jobs[run_id] = {
                "status": "failed",
                "run_id": run_id,
                "n_sampled": 0,
                "n_judged": 0,
                "n_failed": 0,
                "avg_score": None,
                "error": str(exc),
            }


@router.post("/eval/run", response_model=dict)
def trigger_eval_run(
    n: int = Query(default=5, ge=1, le=20),
    max_age_days: int = Query(default=7, ge=1, le=90),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Start an async eval run and return immediately.

    Poll GET /eval/run/status/{run_id} to check progress.
    """
    run_id = str(uuid.uuid4())[:8]
    with _eval_jobs_lock:
        _eval_jobs[run_id] = {"status": "running", "run_id": run_id}
    background_tasks.add_task(_do_eval_run, run_id, n, max_age_days)
    return {"run_id": run_id, "status": "running"}


@router.get("/eval/run/status/{run_id}")
def eval_run_status(run_id: str):
    """Poll for the result of a previously triggered eval run."""
    with _eval_jobs_lock:
        job = _eval_jobs.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No eval run with id '{run_id}'")
    return job


@router.get("/eval/run/history", response_model=EvalRunHistoryResponse)
def eval_run_history(limit: int = Query(default=20, ge=1, le=200)):
    """Return the most recent eval-run summaries."""
    raw = _eval_pipeline.run_history(limit=limit)
    return EvalRunHistoryResponse(entries=[EvalRunHistoryEntry(**r) for r in raw])
