from __future__ import annotations

import asyncio
import json
import random
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Literal, cast
from app.core.models import ChatRequest, ChatResponse, ConversationHistoryResponse, ToolResult, TurnOut
from app.core.config import settings
from app.agent.conversation import ConversationStore, Turn
from app.agent.judge_metrics import JudgeRecord, judge_metrics
from app.agent.planner import Planner
from app.agent.executor import Executor
from app.agent.llm import LLMReasoner, LLMUnavailable

router = APIRouter()
planner = Planner()
executor = Executor()
reasoner = LLMReasoner()
conversations = ConversationStore()
conversations.evict_old()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    conversation_id = req.conversation_id or str(uuid.uuid4())
    conversation = conversations.get_or_create(conversation_id)
    dataset_id = req.dataset_id or conversation.last_dataset_id
    history = conversation.recent_history()

    loop = asyncio.get_running_loop()

    _plan_result = await loop.run_in_executor(
        None,
        lambda: planner.plan(
            req.message,
            dataset_id,
            top_k=req.top_k,
            conversation_history=history,
            trained_model_ids=conversation.trained_model_ids,
        ),
    )
    tool_calls, citations, _ps, llm_error, llm_notes = _plan_result
    planning_source: Literal["llm", "rules"] = cast(Literal["llm", "rules"], _ps)

    tool_results: list[ToolResult] = []
    tables: list[dict] = []
    charts: list[dict] = []

    if dataset_id and tool_calls:
        try:
            tool_results, tables, charts = await loop.run_in_executor(
                None, executor.run, dataset_id, tool_calls
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            message = f"Tool execution failed: {e}"
            _record_turn(conversation, req.message, message, dataset_id, tool_calls, tool_results)
            conversations.save(conversation)
            return ChatResponse(
                dataset_id=dataset_id,
                conversation_id=conversation_id,
                message=message,
                tool_calls=tool_calls,
                tool_results=tool_results,
                tables=tables,
                charts=charts,
                citations=citations,
                llm_enabled=reasoner.enabled,
                planning_source=planning_source,
                llm_error=llm_error,
                llm_notes=llm_notes,
            )

    dataset_context = None
    if reasoner.enabled and dataset_id:
        try:
            df_sample = await loop.run_in_executor(
                None, executor.dm.load_df, dataset_id, settings.llm_analysis_sample_rows
            )
            dataset_context = reasoner.dataset_analysis_context(df_sample)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception:
            dataset_context = None

    msg_lines = []
    if citations:
        msg_lines.append("RAG context pulled from your analytics corpus to guide the approach.")
    if dataset_id:
        msg_lines.append(f"Dataset: {dataset_id}")
    if tool_calls:
        msg_lines.append(f"Planned tools: {', '.join(tc.name for tc in tool_calls)}")

    message = "\n".join(msg_lines) if msg_lines else "OK"
    synthesis_source: Literal["llm", "rules"] = "rules"
    if reasoner.enabled:
        try:
            message = await loop.run_in_executor(
                None,
                lambda: reasoner.synthesize(
                    req.message,
                    dataset_id=dataset_id,
                    dataset_context=dataset_context,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    citations=citations,
                    conversation_history=history,
                ),
            )
            synthesis_source = "llm"
        except LLMUnavailable as e:
            llm_error = str(e)
            if message == "OK":
                message = f"LLM synthesis unavailable: {e}"
            else:
                message = f"{message}\nLLM synthesis unavailable: {e}"

    _record_turn(conversation, req.message, message, dataset_id, tool_calls, tool_results)
    conversations.save(conversation)

    groundedness_score = None
    groundedness_issues: list[str] = []
    if synthesis_source == "llm" and random.random() < settings.llm_judge_sample_rate:
        try:
            verdict = await loop.run_in_executor(
                None,
                lambda: reasoner.judge_groundedness(
                    message, dataset_context=dataset_context, tool_results=tool_results
                ),
            )
            groundedness_score = verdict["score"]
            groundedness_issues = verdict["issues"]
            judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])))
        except LLMUnavailable:
            pass

    return ChatResponse(
        dataset_id=dataset_id,
        conversation_id=conversation_id,
        message=message,
        tool_calls=tool_calls,
        tool_results=tool_results,
        tables=tables,
        charts=charts,
        citations=citations,
        llm_enabled=reasoner.enabled,
        planning_source=planning_source,
        synthesis_source=synthesis_source,
        llm_error=llm_error,
        llm_notes=llm_notes,
        groundedness_score=groundedness_score,
        groundedness_issues=groundedness_issues,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint — yields tool results as they complete, then the final synthesis."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    async def generate():
        loop = asyncio.get_running_loop()
        conversation_id = req.conversation_id or str(uuid.uuid4())
        conversation = conversations.get_or_create(conversation_id)
        dataset_id = req.dataset_id or conversation.last_dataset_id
        history = conversation.recent_history()

        _plan_result = await loop.run_in_executor(
            None,
            lambda: planner.plan(
                req.message,
                dataset_id,
                top_k=req.top_k,
                conversation_history=history,
                trained_model_ids=conversation.trained_model_ids,
            ),
        )
        tool_calls, citations, _ps, llm_error, llm_notes = _plan_result
        planning_source: Literal["llm", "rules"] = cast(Literal["llm", "rules"], _ps)

        yield f"data: {json.dumps({'type': 'plan', 'tool_calls': [tc.model_dump() for tc in tool_calls], 'conversation_id': conversation_id})}\n\n"

        all_tool_results, all_tables, all_charts = [], [], []

        if dataset_id and tool_calls:
            try:
                # Collect the sync generator in a thread so the event loop stays free.
                collected = await loop.run_in_executor(
                    None, lambda: list(executor.run_stream(dataset_id, tool_calls))
                )
                for tool_result, tables, charts in collected:
                    all_tool_results.append(tool_result)
                    all_tables.extend(tables)
                    all_charts.extend(charts)
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_result.name, 'ok': tool_result.ok, 'error': tool_result.error})}\n\n"
            except KeyError as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                return
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                return

        dataset_context = None
        if reasoner.enabled and dataset_id:
            try:
                df_sample = await loop.run_in_executor(
                    None, executor.dm.load_df, dataset_id, settings.llm_analysis_sample_rows
                )
                dataset_context = reasoner.dataset_analysis_context(df_sample)
            except Exception:
                pass

        msg_parts = []
        if citations:
            msg_parts.append("RAG context pulled from your analytics corpus to guide the approach.")
        if dataset_id:
            msg_parts.append(f"Dataset: {dataset_id}")
        if tool_calls:
            msg_parts.append(f"Planned tools: {', '.join(tc.name for tc in tool_calls)}")
        message = "\n".join(msg_parts) if msg_parts else "OK"

        synthesis_source: Literal["llm", "rules"] = "rules"
        if reasoner.enabled:
            try:
                message = await loop.run_in_executor(
                    None,
                    lambda: reasoner.synthesize(
                        req.message,
                        dataset_id=dataset_id,
                        dataset_context=dataset_context,
                        tool_calls=tool_calls,
                        tool_results=all_tool_results,
                        citations=citations,
                        conversation_history=history,
                    ),
                )
                synthesis_source = "llm"
            except LLMUnavailable as e:
                llm_error = str(e)

        _record_turn(conversation, req.message, message, dataset_id, tool_calls, all_tool_results)
        conversations.save(conversation)

        groundedness_score = None
        groundedness_issues: list[str] = []
        if synthesis_source == "llm" and random.random() < settings.llm_judge_sample_rate:
            try:
                verdict = await loop.run_in_executor(
                    None,
                    lambda: reasoner.judge_groundedness(
                        message, dataset_context=dataset_context, tool_results=all_tool_results
                    ),
                )
                groundedness_score = verdict["score"]
                groundedness_issues = verdict["issues"]
                judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])))
            except LLMUnavailable:
                pass

        final = ChatResponse(
            dataset_id=dataset_id,
            conversation_id=conversation_id,
            message=message,
            tool_calls=tool_calls,
            tool_results=all_tool_results,
            tables=all_tables,
            charts=all_charts,
            citations=citations,
            llm_enabled=reasoner.enabled,
            planning_source=planning_source,
            synthesis_source=synthesis_source,
            llm_error=llm_error,
            llm_notes=llm_notes,
            groundedness_score=groundedness_score,
            groundedness_issues=groundedness_issues,
        )
        yield f"data: {json.dumps({'type': 'done', 'response': final.model_dump()})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/{conversation_id}/history", response_model=ConversationHistoryResponse)
def get_history(conversation_id: str):
    conversation = conversations.get_or_create(conversation_id)
    turns = [
        TurnOut(role=t.role, content=t.content, dataset_id=t.dataset_id, tool_calls=t.tool_calls, timestamp=t.timestamp)
        for t in conversation.turns
    ]
    return ConversationHistoryResponse(conversation_id=conversation_id, turns=turns)


def _record_turn(conversation, user_message, assistant_message, dataset_id, tool_calls, tool_results) -> None:
    conversation.add_turn(Turn(role="user", content=user_message, dataset_id=dataset_id))
    conversation.add_turn(
        Turn(
            role="assistant",
            content=assistant_message,
            dataset_id=dataset_id,
            tool_calls=[tc.model_dump() for tc in tool_calls],
        )
    )
    if dataset_id:
        conversation.last_dataset_id = dataset_id

    for result in tool_results:
        if result.name == "train_supervised_model" and result.ok and isinstance(result.result, dict):
            model_id = result.result.get("model_id")
            if model_id and model_id not in conversation.trained_model_ids:
                conversation.trained_model_ids.append(model_id)
