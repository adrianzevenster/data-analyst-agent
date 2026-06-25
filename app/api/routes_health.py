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
    RagEvalQueryResult,
    RagEvalResponse,
    RepairStatsResponse,
    ScoringLatencyResponse,
    ScoringModelLatency,
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


@router.get("/health/scoring-latency", response_model=ScoringLatencyResponse)
def scoring_latency_health():
    from app.agent.latency_metrics import scoring_latency
    raw = scoring_latency.snapshot()
    return ScoringLatencyResponse(
        n_models=raw["n_models"],
        by_model={mid: ScoringModelLatency(**stats) for mid, stats in raw["by_model"].items()},
    )


@router.get("/health/rag-eval", response_model=RagEvalResponse)
def rag_eval_health():
    report_path = settings.eval_path / "rag_retrieval_eval.json"
    if not report_path.exists():
        return RagEvalResponse(available=False)
    try:
        report = json.loads(report_path.read_text())
        return RagEvalResponse(
            available=True,
            n_queries=report.get("n_queries", 0),
            recall_at_1=report.get("recall_at_1"),
            recall_at_3=report.get("recall_at_3"),
            recall_at_5=report.get("recall_at_5"),
            mrr=report.get("mrr"),
            generated_at=report.get("generated_at"),
            queries=[RagEvalQueryResult(**q) for q in report.get("queries", [])],
        )
    except Exception:
        return RagEvalResponse(available=False)


@router.post("/health/rag-eval/run", response_model=RagEvalResponse)
def run_rag_eval(background_tasks: BackgroundTasks = BackgroundTasks()):
    """Generate a live self-recall eval from the current FAISS index."""
    import pathlib
    import re
    import time

    def _do() -> None:
        try:
            from app.rag.retriever import RagRetriever
            from app.rag.store import FaissStore

            store = FaissStore(index_dir=str(settings.index_dir))
            all_chunks = store.chunks if hasattr(store, "chunks") else []
            if not all_chunks:
                return

            # Group chunks by source
            from collections import defaultdict
            by_source: dict[str, list] = defaultdict(list)
            for ch in all_chunks:
                by_source[ch.source_id.split("#")[0]].append(ch)

            def _best_query(chunks: list) -> str | None:
                """Pick the longest, most prose-like sentence from up to 5 chunks."""
                for ch in chunks[:5]:
                    for sent in re.split(r"[.!?\n]", ch.text.strip()):
                        sent = sent.strip()
                        # Reject short, all-caps, or obviously non-prose strings
                        if len(sent) < 30 or len(sent) > 200:
                            continue
                        words = sent.split()
                        if len(words) < 5:
                            continue
                        alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
                        if alpha_ratio < 0.5:
                            continue
                        return sent
                return None

            query_pairs: list[tuple[str, str]] = []
            for src, chunks in by_source.items():
                q = _best_query(chunks)
                if q:
                    query_pairs.append((q, src))
                if len(query_pairs) >= 15:
                    break

            if not query_pairs:
                return

            retriever = RagRetriever()
            TOP_K = 5
            query_results = []
            rr_sum = 0.0
            hit1 = hit3 = hit5 = 0

            for q_text, expected_src in query_pairs:
                hits = retriever.retrieve(q_text, top_k=TOP_K)
                top_sources = [
                    h.get("source_id", "") if isinstance(h, dict) else str(h)
                    for h in hits
                ]
                top_score = hits[0].get("score") if hits and isinstance(hits[0], dict) else None

                # Find rank of first matching hit (1-based)
                rank = None
                for i, src in enumerate(top_sources, 1):
                    if expected_src in src:
                        rank = i
                        break

                hit = rank is not None
                if rank is not None:
                    rr_sum += 1.0 / rank
                    if rank <= 1:
                        hit1 += 1
                    if rank <= 3:
                        hit3 += 1
                    hit5 += 1

                query_results.append({
                    "query": q_text,
                    "expected_source": expected_src,
                    "hit": hit,
                    "rank": rank,
                    "top_sources": [s.split("#")[0] for s in top_sources[:3]],
                    "score": round(top_score, 4) if top_score is not None else None,
                })

            n = len(query_pairs)
            report = {
                "n_queries": n,
                "recall_at_1": round(hit1 / n, 3),
                "recall_at_3": round(hit3 / n, 3),
                "recall_at_5": round(hit5 / n, 3),
                "mrr": round(rr_sum / n, 3),
                "generated_at": time.time(),
                "queries": query_results,
            }
            out = pathlib.Path(settings.eval_path) / "rag_retrieval_eval.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("rag-eval/run failed: %s", exc)

    background_tasks.add_task(_do)
    return RagEvalResponse(available=False, n_queries=0)


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
        from app.core.config import settings
        judge_timeout = max(60, getattr(settings, "llm_timeout_seconds", 120))
        result = _eval_pipeline.run(reasoner=reasoner, dm=dm, n=n, max_age_days=max_age_days, judge_timeout=judge_timeout)
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
