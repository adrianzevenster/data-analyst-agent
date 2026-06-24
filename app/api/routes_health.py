import json

from fastapi import APIRouter, Query

from app.agent.judge_metrics import judge_metrics
from app.agent.latency_metrics import latency_metrics
from app.agent.llm_metrics import metrics
from app.analytics.quality_eval import QualityEvalPipeline
from app.core.config import settings
from app.core.models import (
    EvalRunHistoryEntry,
    EvalRunHistoryResponse,
    EvalRunResult,
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

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


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


@router.post("/eval/run", response_model=EvalRunResult)
def trigger_eval_run(
    n: int = Query(default=20, ge=1, le=100),
    max_age_days: int = Query(default=7, ge=1, le=90),
):
    """Sample recent conversation turns and judge them with the LLM.

    Results are written to judge_log (visible in /health/quality-trend) and
    eval_run_log (visible in /eval/run/history). No-ops gracefully when LLM
    is unavailable — returns n_judged=0.
    """
    from app.agent.llm import LLMReasoner
    from app.agent.executor import Executor

    reasoner = LLMReasoner()
    dm = Executor().dm
    result = _eval_pipeline.run(reasoner=reasoner, dm=dm, n=n, max_age_days=max_age_days)
    return EvalRunResult(**result)


@router.get("/eval/run/history", response_model=EvalRunHistoryResponse)
def eval_run_history(limit: int = Query(default=20, ge=1, le=200)):
    """Return the most recent eval-run summaries."""
    raw = _eval_pipeline.run_history(limit=limit)
    return EvalRunHistoryResponse(entries=[EvalRunHistoryEntry(**r) for r in raw])
