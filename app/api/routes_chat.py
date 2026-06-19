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

_TRAIN_KEYWORDS = ["train", "build a model", "build model", "fit a model", "train model"]


def _rule_message(
    user_message: str,
    tool_calls: list,
    dataset_id: str | None,
    citations: list,
) -> str:
    """Generate a readable rule-based message without LLM synthesis."""
    m = user_message.lower()

    # Train requested but no train call planned → tell user to name a target column
    train_requested = any(k in m for k in _TRAIN_KEYWORDS)
    has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
    if train_requested and not has_train_call:
        if dataset_id:
            try:
                df = executor.dm.load_df(dataset_id, limit=5)
                cols = ", ".join(f"**{c}**" for c in df.columns)
                return (
                    "To train a model I need to know which column to predict. "
                    f"Your dataset has these columns: {cols}.\n\n"
                    "Please specify the target — for example: "
                    "*'Train a model to predict debit'*"
                )
            except Exception:
                pass
        return (
            "To train a model, please specify the target column — "
            "for example: *'Train a model to predict revenue'*"
        )

    # Generic readable summary
    parts: list[str] = []
    if tool_calls:
        names = ", ".join(tc.name for tc in tool_calls)
        parts.append(f"Ran: {names}.")
    if citations:
        parts.append("Relevant analytics guidance was retrieved from the corpus.")
    return " ".join(parts) if parts else "Done."


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

    message = _rule_message(req.message, tool_calls, dataset_id, citations)
    synthesis_source: Literal["llm", "rules"] = "rules"

    # Skip LLM synthesis when a deterministic template applies (e.g. train-without-target)
    train_requested = any(k in req.message.lower() for k in _TRAIN_KEYWORDS)
    has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
    skip_llm = train_requested and not has_train_call

    if reasoner.enabled and not skip_llm:
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
        except Exception as e:
            llm_error = str(e)

    groundedness_score = None
    groundedness_criteria: dict[str, int] = {}
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
            groundedness_criteria = verdict.get("criteria", {})
            groundedness_issues = verdict["issues"]
            judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])))
        except LLMUnavailable:
            pass

    _record_turn(
        conversation, req.message, message, dataset_id, tool_calls, tool_results,
        tables=tables, charts=charts,
        groundedness_score=groundedness_score,
        groundedness_criteria=groundedness_criteria,
        groundedness_issues=groundedness_issues,
        planning_source=planning_source,
        synthesis_source=synthesis_source,
    )
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
        synthesis_source=synthesis_source,
        llm_error=llm_error,
        llm_notes=llm_notes,
        groundedness_score=groundedness_score,
        groundedness_criteria=groundedness_criteria,
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

        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

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

        message = _rule_message(req.message, tool_calls, dataset_id, citations)
        synthesis_source: Literal["llm", "rules"] = "rules"

        train_requested = any(k in req.message.lower() for k in _TRAIN_KEYWORDS)
        has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
        skip_llm = train_requested and not has_train_call

        if reasoner.enabled and not skip_llm:
            yield f"data: {json.dumps({'type': 'synthesizing'})}\n\n"
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
            except Exception as e:
                llm_error = str(e)

        groundedness_score = None
        groundedness_criteria: dict[str, int] = {}
        groundedness_issues: list[str] = []
        if synthesis_source == "llm" and random.random() < settings.llm_judge_sample_rate:
            yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
            try:
                verdict = await loop.run_in_executor(
                    None,
                    lambda: reasoner.judge_groundedness(
                        message, dataset_context=dataset_context, tool_results=all_tool_results
                    ),
                )
                groundedness_score = verdict["score"]
                groundedness_criteria = verdict.get("criteria", {})
                groundedness_issues = verdict["issues"]
                judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])))
            except LLMUnavailable:
                pass

        _record_turn(
            conversation, req.message, message, dataset_id, tool_calls, all_tool_results,
            tables=all_tables, charts=all_charts,
            groundedness_score=groundedness_score,
            groundedness_criteria=groundedness_criteria,
            groundedness_issues=groundedness_issues,
            planning_source=planning_source,
            synthesis_source=synthesis_source,
        )
        conversations.save(conversation)

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
            groundedness_criteria=groundedness_criteria,
            groundedness_issues=groundedness_issues,
        )
        try:
            payload = json.dumps({"type": "done", "response": final.model_dump()})
        except (TypeError, ValueError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Response serialization failed: {exc}'})}\n\n"
            return
        yield f"data: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conversation_id}/history", response_model=ConversationHistoryResponse)
def get_history(conversation_id: str):
    conversation = conversations.get_or_create(conversation_id)
    turns = [
        TurnOut(
            role=t.role,
            content=t.content,
            dataset_id=t.dataset_id,
            tool_calls=t.tool_calls,
            timestamp=t.timestamp,
            tables=t.tables,
            charts=t.charts,
            groundedness_score=t.groundedness_score,
            groundedness_criteria=t.groundedness_criteria,
            groundedness_issues=t.groundedness_issues,
            planning_source=t.planning_source,
            synthesis_source=t.synthesis_source,
        )
        for t in conversation.turns
    ]
    return ConversationHistoryResponse(conversation_id=conversation_id, turns=turns)


def _record_turn(
    conversation,
    user_message,
    assistant_message,
    dataset_id,
    tool_calls,
    tool_results,
    tables=None,
    charts=None,
    groundedness_score=None,
    groundedness_criteria=None,
    groundedness_issues=None,
    planning_source="rules",
    synthesis_source="rules",
) -> None:
    conversation.add_turn(Turn(role="user", content=user_message, dataset_id=dataset_id))
    conversation.add_turn(
        Turn(
            role="assistant",
            content=assistant_message,
            dataset_id=dataset_id,
            tool_calls=[tc.model_dump() for tc in tool_calls],
            tables=tables or [],
            charts=charts or [],
            groundedness_score=groundedness_score,
            groundedness_criteria=groundedness_criteria or {},
            groundedness_issues=groundedness_issues or [],
            planning_source=planning_source,
            synthesis_source=synthesis_source,
        )
    )
    if dataset_id:
        conversation.last_dataset_id = dataset_id

    for result in tool_results:
        if result.name == "train_supervised_model" and result.ok and isinstance(result.result, dict):
            model_id = result.result.get("model_id")
            if model_id and model_id not in conversation.trained_model_ids:
                conversation.trained_model_ids.append(model_id)
