import json

from fastapi import APIRouter

from app.agent.judge_metrics import judge_metrics
from app.agent.llm_metrics import metrics
from app.core.config import settings
from app.core.models import JudgeHistoryResponse, JudgeStatsResponse, LLMStatsResponse, RagEvalResponse, RepairStatsResponse

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
    entries = judge_metrics.history(limit=min(limit, 2000))
    return JudgeHistoryResponse(entries=entries, total=len(entries))


@router.get("/health/llm-repair", response_model=RepairStatsResponse)
def llm_repair_health():
    return metrics.repair_snapshot()


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
