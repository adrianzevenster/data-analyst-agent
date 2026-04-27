from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.models import ChatRequest, ChatResponse
from app.agent.planner import Planner
from app.agent.executor import Executor

router = APIRouter()
planner = Planner()
executor = Executor()


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

    msg_lines = []
    if citations:
        msg_lines.append("RAG context pulled from your analytics corpus to guide the approach.")
    if req.dataset_id:
        msg_lines.append(f"Dataset: {req.dataset_id}")
    if tool_calls:
        msg_lines.append(f"Planned tools: {', '.join(tc.name for tc in tool_calls)}")

    return ChatResponse(
        dataset_id=req.dataset_id,
        message="\n".join(msg_lines) if msg_lines else "OK",
        tool_calls=tool_calls,
        tool_results=tool_results,
        tables=tables,
        charts=charts,
        citations=citations,
    )
