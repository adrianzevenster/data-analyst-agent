from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.models import ChatRequest, ChatResponse
from app.core.config import settings
from app.agent.planner import Planner
from app.agent.executor import Executor
from app.agent.llm import LLMReasoner, LLMUnavailable

router = APIRouter()
planner = Planner()
executor = Executor()
reasoner = LLMReasoner()


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    tool_calls, citations = planner.plan(req.message, req.dataset_id, top_k=req.top_k)

    tool_results = []
    tables = []
    charts = []

    if req.dataset_id and tool_calls:
        try:
            tool_results, tables, charts = executor.run(req.dataset_id, tool_calls)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            return ChatResponse(
                dataset_id=req.dataset_id,
                message=f"Tool execution failed: {e}",
                tool_calls=tool_calls,
                tool_results=tool_results,
                tables=tables,
                charts=charts,
                citations=citations,
            )

    dataset_context = None
    if reasoner.enabled and req.dataset_id:
        try:
            df_sample = executor.dm.load_df(req.dataset_id, limit=settings.llm_analysis_sample_rows)
            dataset_context = reasoner.dataset_analysis_context(df_sample)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception:
            dataset_context = None

    msg_lines = []
    if citations:
        msg_lines.append("RAG context pulled from your analytics corpus to guide the approach.")
    if req.dataset_id:
        msg_lines.append(f"Dataset: {req.dataset_id}")
    if tool_calls:
        msg_lines.append(f"Planned tools: {', '.join(tc.name for tc in tool_calls)}")

    message = "\n".join(msg_lines) if msg_lines else "OK"
    if reasoner.enabled:
        try:
            message = reasoner.synthesize(
                req.message,
                dataset_id=req.dataset_id,
                dataset_context=dataset_context,
                tool_calls=tool_calls,
                tool_results=tool_results,
                citations=citations,
            )
        except LLMUnavailable as e:
            if message == "OK":
                message = f"LLM synthesis unavailable: {e}"
            else:
                message = f"{message}\nLLM synthesis unavailable: {e}"

    return ChatResponse(
        dataset_id=req.dataset_id,
        message=message,
        tool_calls=tool_calls,
        tool_results=tool_results,
        tables=tables,
        charts=charts,
        citations=citations,
    )
