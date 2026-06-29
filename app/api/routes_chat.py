from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Literal, cast
from app.core.models import ChatRequest, ChatResponse, ConversationHistoryResponse, JudgeStatus, ToolResult, TurnOut
from app.core.config import settings
from app.agent.conversation import ConversationStore, Turn
from app.agent.judge_metrics import JudgeRecord, judge_metrics
from app.agent.latency_metrics import latency_metrics
from app.agent.planner import Planner
from app.agent.executor import Executor
from app.agent.llm import LLMReasoner, LLMUnavailable

logger = logging.getLogger(__name__)

router = APIRouter()
planner = Planner()
executor = Executor()
reasoner = LLMReasoner()
conversations = ConversationStore()
conversations.evict_old()

# Match "train" as a whole word, or other explicit training phrases.
# \btrain\b does NOT match "trained", "retraining", "constraint", etc.
_TRAIN_RE = re.compile(
    r'\btrain\b|build\s+(?:a\s+)?model|fit\s+(?:a\s+)?model',
    re.IGNORECASE,
)


def _is_train_requested(message: str) -> bool:
    return bool(_TRAIN_RE.search(message))


def _target_example_column(cols) -> str:
    priority_terms = (
        "target",
        "label",
        "actual",
        "total_fare",
        "fare",
        "revenue",
        "sales",
        "income",
        "price",
        "cost",
        "amount",
    )
    col_names = [str(c) for c in cols]
    for term in priority_terms:
        match = next((c for c in col_names if term in c.lower()), None)
        if match:
            return match
    return col_names[0] if col_names else "target_column"


def _rule_judge_status() -> JudgeStatus:
    return "llm_disabled" if not reasoner.enabled else "rule_based"


def _slim_results_for_continuation(tool_results: list) -> list[dict]:
    """Compact summary of tool results passed to the LLM for the agentic continuation step."""
    slim = []
    for tr in tool_results:
        entry: dict = {"tool": tr.name, "ok": tr.ok}
        if not tr.ok:
            entry["error"] = tr.error
        elif isinstance(tr.result, dict):
            readout = tr.result.get("engineering_readout") or tr.result.get("readout", "")
            if readout:
                entry["summary"] = str(readout)[:400]
            for key in ("model_id", "task_type", "target_col", "n_anomalies", "n_clusters"):
                if key in tr.result and isinstance(tr.result[key], (str, int, float)):
                    entry[key] = tr.result[key]
        slim.append(entry)
    return slim


def _should_sample_judge() -> bool:
    rate = settings.llm_judge_sample_rate
    if rate <= 0:
        return False
    if rate >= 1:
        return True

    snapshot = judge_metrics.snapshot()
    if snapshot["attempted_count"] == 0 and snapshot["sampled_count"] == 0:
        return True

    return random.random() < rate


def _rule_message(
    user_message: str,
    tool_calls: list,
    dataset_id: str | None,
    citations: list,
    clarification: str | None = None,
) -> str:
    """Generate a readable rule-based message without LLM synthesis."""
    # Clarification signal from planner: the intent was ambiguous.
    if clarification:
        return clarification

    # Train requested but no train call planned → tell user to name a target column.
    train_requested = _is_train_requested(user_message)
    has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
    if train_requested and not has_train_call:
        if dataset_id:
            try:
                df = executor.dm.load_df(dataset_id, limit=5)
                cols = ", ".join(f"**{c}**" for c in df.columns)
                example_col = _target_example_column(df.columns)
                return (
                    "To train a model I need to know which column to predict. "
                    f"Your dataset has these columns: {cols}.\n\n"
                    "Please specify the target — for example: "
                    f"*'Train a model to predict {example_col}'*"
                )
            except Exception:
                pass
        return (
            "To train a model, please specify the target column — "
            "for example: *'Train a model to predict revenue'*"
        )

    # Generic readable summary.
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
    # Rich history (with prior tool results) for planning; plain text for synthesis.
    plan_history = conversation.recent_history_with_tool_context()
    history = conversation.recent_history()

    loop = asyncio.get_running_loop()

    _t0_plan = time.perf_counter()
    _plan_result = await loop.run_in_executor(
        None,
        lambda: planner.plan(
            req.message,
            dataset_id,
            top_k=req.top_k,
            conversation_history=plan_history,
            trained_model_ids=conversation.trained_model_ids,
        ),
    )
    _t_plan_ms = (time.perf_counter() - _t0_plan) * 1000
    tool_calls, citations, _ps, llm_error, llm_notes = _plan_result
    planning_source: Literal["llm", "rules"] = cast(Literal["llm", "rules"], _ps)

    # Separate the clarify sentinel (never executed) from real tool calls.
    _clarify = next((tc for tc in tool_calls if tc.name == "clarify"), None)
    tool_calls = [tc for tc in tool_calls if tc.name != "clarify"]
    _clarification: str | None = _clarify.arguments.get("question") if _clarify else None

    tool_results: list[ToolResult] = []
    tables: list[dict] = []
    charts: list[dict] = []
    _t_execute_ms = 0.0

    if dataset_id and tool_calls:
        try:
            _t0_exec = time.perf_counter()
            tool_results, tables, charts = await loop.run_in_executor(
                None, executor.run, dataset_id, tool_calls
            )
            _t_execute_ms = (time.perf_counter() - _t0_exec) * 1000
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            message = f"Tool execution failed: {e}"
            judge_status = _rule_judge_status()
            judge_metrics.record_skipped(judge_status)
            _record_turn(
                conversation,
                req.message,
                message,
                dataset_id,
                tool_calls,
                tool_results,
                judge_status=judge_status,
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
                llm_error=llm_error,
                llm_notes=llm_notes,
                judge_status=judge_status,
            )

        # Agentic continuation: give the LLM its step results and ask what to do next.
        if (
            reasoner.enabled
            and settings.llm_agentic_continuation
            and tool_results
            and planning_source == "llm"
        ):
            try:
                ran_names = {tc.name for tc in tool_calls}
                prior_slim = _slim_results_for_continuation(tool_results)
                cont_calls, _, _, _, cont_notes = await loop.run_in_executor(
                    None,
                    lambda: planner.plan(
                        req.message,
                        dataset_id,
                        conversation_history=plan_history,
                        trained_model_ids=conversation.trained_model_ids,
                        prior_step_results=prior_slim,
                    ),
                )
                # Only execute tools not already run this turn.
                cont_calls = [c for c in cont_calls if c.name not in ran_names]
                if cont_calls:
                    _t0_cont = time.perf_counter()
                    cont_results, cont_tables, cont_charts = await loop.run_in_executor(
                        None, executor.run, dataset_id, cont_calls
                    )
                    _t_execute_ms += (time.perf_counter() - _t0_cont) * 1000
                    tool_calls = list(tool_calls) + cont_calls
                    tool_results = list(tool_results) + cont_results
                    tables = list(tables) + cont_tables
                    charts = list(charts) + cont_charts
                    llm_notes = list(llm_notes) + cont_notes
            except Exception:
                pass  # continuation failure never blocks the main response

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

    message = _rule_message(req.message, tool_calls, dataset_id, citations, clarification=_clarification)
    synthesis_source: Literal["llm", "rules"] = "rules"

    # Skip LLM synthesis when: (a) training was requested but didn't happen and
    # there are no tool results to synthesise, or (b) the planner issued a
    # clarification question — synthesising over it would reword or lose it.
    train_requested = _is_train_requested(req.message)
    has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
    skip_llm = (_clarification is not None) or (train_requested and not has_train_call and not tool_results)

    _t0_synth = time.perf_counter()
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
    _t_synthesis_ms = (time.perf_counter() - _t0_synth) * 1000

    groundedness_score = None
    groundedness_criteria: dict[str, int] = {}
    groundedness_issues: list[str] = []
    judge_status = _rule_judge_status()
    if reasoner.enabled:
        if _should_sample_judge():
            judge_status = "failed"
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
                judge_status = "judged"
                judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])), synthesis_source=synthesis_source)
            except LLMUnavailable as e:
                logger.warning("LLM judge failed: %s", e)
                judge_metrics.record_failure(str(e))
            except Exception as e:
                logger.exception("LLM judge failed unexpectedly: %s", e)
                judge_metrics.record_failure(str(e))
        else:
            judge_status = "not_sampled"
            judge_metrics.record_skipped(judge_status)
    else:
        judge_metrics.record_skipped(judge_status)

    _latency = {"t_plan": round(_t_plan_ms, 1), "t_execute": round(_t_execute_ms, 1), "t_synthesis": round(_t_synthesis_ms, 1)}
    latency_metrics.record(_t_plan_ms, _t_execute_ms, _t_synthesis_ms)
    _record_turn(
        conversation, req.message, message, dataset_id, tool_calls, tool_results,
        tables=tables, charts=charts, citations=citations,
        groundedness_score=groundedness_score,
        groundedness_criteria=groundedness_criteria,
        groundedness_issues=groundedness_issues,
        judge_status=judge_status,
        planning_source=planning_source,
        synthesis_source=synthesis_source,
        latency_ms=_latency,
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
        judge_status=judge_status,
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
        plan_history = conversation.recent_history_with_tool_context()
        history = conversation.recent_history()

        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        _t0_plan = time.perf_counter()
        try:
            _plan_future = loop.run_in_executor(
                None,
                lambda: planner.plan(
                    req.message,
                    dataset_id,
                    top_k=req.top_k,
                    conversation_history=plan_history,
                    trained_model_ids=conversation.trained_model_ids,
                ),
            )
            while True:
                try:
                    _plan_result = await asyncio.wait_for(asyncio.shield(_plan_future), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
            tool_calls, citations, _ps, llm_error, llm_notes = _plan_result
        except Exception as e:
            logger.exception("Planning failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Planning failed: {e}'})}\n\n"
            return
        _t_plan_ms = (time.perf_counter() - _t0_plan) * 1000
        planning_source: Literal["llm", "rules"] = cast(Literal["llm", "rules"], _ps)

        # Separate the clarify sentinel (never executed) from real tool calls.
        _clarify = next((tc for tc in tool_calls if tc.name == "clarify"), None)
        tool_calls = [tc for tc in tool_calls if tc.name != "clarify"]
        _clarification: str | None = _clarify.arguments.get("question") if _clarify else None

        logger.info("stream: plan ok tool_calls=%d", len(tool_calls))

        try:
            yield f"data: {json.dumps({'type': 'plan', 'tool_calls': [tc.model_dump() for tc in tool_calls], 'conversation_id': conversation_id})}\n\n"
        except Exception as e:
            logger.exception("Plan serialization failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Plan serialization failed: {e}'})}\n\n"
            return

        all_tool_results, all_tables, all_charts = [], [], []
        _t_execute_ms = 0.0

        if dataset_id and tool_calls:
            try:
                _t0_exec = time.perf_counter()
                # Stream tool results as each tool completes via a thread-safe queue
                # so SSE events flow in real-time (no silent gap during multi-tool EDA runs).
                _tool_q: asyncio.Queue = asyncio.Queue()

                def _run_tools() -> None:
                    try:
                        for item in executor.run_stream(dataset_id, tool_calls):
                            loop.call_soon_threadsafe(_tool_q.put_nowait, item)
                    except Exception as _e:
                        loop.call_soon_threadsafe(_tool_q.put_nowait, _e)
                    finally:
                        loop.call_soon_threadsafe(_tool_q.put_nowait, None)

                loop.run_in_executor(None, _run_tools)

                while True:
                    try:
                        item = await asyncio.wait_for(_tool_q.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    tool_result, tables, charts = item
                    all_tool_results.append(tool_result)
                    all_tables.extend(tables)
                    all_charts.extend(charts)
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_result.name, 'ok': tool_result.ok, 'error': tool_result.error})}\n\n"

                _t_execute_ms = (time.perf_counter() - _t0_exec) * 1000

                # Agentic continuation: one follow-up planning step with step results.
                if (
                    reasoner.enabled
                    and settings.llm_agentic_continuation
                    and all_tool_results
                    and planning_source == "llm"
                ):
                    try:
                        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
                        ran_names = {tc.name for tc in tool_calls}
                        prior_slim = _slim_results_for_continuation(all_tool_results)
                        cont_future = loop.run_in_executor(
                            None,
                            lambda: planner.plan(
                                req.message,
                                dataset_id,
                                conversation_history=plan_history,
                                trained_model_ids=conversation.trained_model_ids,
                                prior_step_results=prior_slim,
                            ),
                        )
                        while True:
                            try:
                                cont_result = await asyncio.wait_for(asyncio.shield(cont_future), timeout=5.0)
                                break
                            except asyncio.TimeoutError:
                                yield ": keepalive\n\n"
                        cont_calls, _, _, _, _ = cont_result
                        cont_calls = [c for c in cont_calls if c.name not in ran_names]
                        if cont_calls:
                            _cont_q: asyncio.Queue = asyncio.Queue()

                            def _run_cont() -> None:
                                try:
                                    for item in executor.run_stream(dataset_id, cont_calls):
                                        loop.call_soon_threadsafe(_cont_q.put_nowait, item)
                                except Exception as _ce:
                                    loop.call_soon_threadsafe(_cont_q.put_nowait, _ce)
                                finally:
                                    loop.call_soon_threadsafe(_cont_q.put_nowait, None)

                            loop.run_in_executor(None, _run_cont)
                            tool_calls = list(tool_calls) + cont_calls
                            while True:
                                try:
                                    citem = await asyncio.wait_for(_cont_q.get(), timeout=5.0)
                                except asyncio.TimeoutError:
                                    yield ": keepalive\n\n"
                                    continue
                                if citem is None:
                                    break
                                if isinstance(citem, Exception):
                                    break
                                ctr, ctables, ccharts = citem
                                all_tool_results.append(ctr)
                                all_tables.extend(ctables)
                                all_charts.extend(ccharts)
                                yield f"data: {json.dumps({'type': 'tool_result', 'name': ctr.name, 'ok': ctr.ok, 'error': ctr.error})}\n\n"
                    except Exception:
                        pass  # continuation failure never blocks the main response

            except KeyError as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                return
            except Exception as e:
                logger.exception("Tool execution failed: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                return

        logger.info("stream: tools done results=%d", len(all_tool_results))
        dataset_context = None
        if reasoner.enabled and dataset_id:
            try:
                df_sample = await loop.run_in_executor(
                    None, executor.dm.load_df, dataset_id, settings.llm_analysis_sample_rows
                )
                dataset_context = reasoner.dataset_analysis_context(df_sample)
            except Exception:
                pass

        message = _rule_message(req.message, tool_calls, dataset_id, citations, clarification=_clarification)
        synthesis_source: Literal["llm", "rules"] = "rules"

        train_requested = _is_train_requested(req.message)
        has_train_call = any(tc.name == "train_supervised_model" for tc in tool_calls)
        skip_llm = (_clarification is not None) or (train_requested and not has_train_call and not all_tool_results)

        _t0_synth = time.perf_counter()
        if reasoner.enabled and not skip_llm:
            yield f"data: {json.dumps({'type': 'synthesizing'})}\n\n"
            try:
                _synth_future = loop.run_in_executor(
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
                while True:
                    try:
                        message = await asyncio.wait_for(asyncio.shield(_synth_future), timeout=5.0)
                        break
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                synthesis_source = "llm"
            except LLMUnavailable as e:
                llm_error = str(e)
            except Exception as e:
                logger.exception("Synthesis failed: %s", e)
                llm_error = str(e)
        _t_synthesis_ms = (time.perf_counter() - _t0_synth) * 1000

        logger.info("stream: synthesis done source=%s", synthesis_source)
        groundedness_score = None
        groundedness_criteria: dict[str, int] = {}
        groundedness_issues: list[str] = []
        judge_status: JudgeStatus = _rule_judge_status()
        if reasoner.enabled:
            if _should_sample_judge():
                judge_status = "failed"
                yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
                try:
                    _judge_future = loop.run_in_executor(
                        None,
                        lambda: reasoner.judge_groundedness(
                            message, dataset_context=dataset_context, tool_results=all_tool_results
                        ),
                    )
                    while True:
                        try:
                            verdict = await asyncio.wait_for(asyncio.shield(_judge_future), timeout=5.0)
                            break
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                    groundedness_score = verdict["score"]
                    groundedness_criteria = verdict.get("criteria", {})
                    groundedness_issues = verdict["issues"]
                    judge_status = "judged"
                    judge_metrics.record(JudgeRecord(score=verdict["score"], issue_count=len(verdict["issues"])), synthesis_source=synthesis_source)
                except LLMUnavailable as e:
                    logger.warning("LLM judge failed: %s", e)
                    judge_metrics.record_failure(str(e))
                except Exception as e:
                    logger.exception("LLM judge failed unexpectedly: %s", e)
                    judge_metrics.record_failure(str(e))
            else:
                judge_status = "not_sampled"
                judge_metrics.record_skipped(judge_status)
        else:
            judge_metrics.record_skipped(judge_status)

        try:
            _latency = {"t_plan": round(_t_plan_ms, 1), "t_execute": round(_t_execute_ms, 1), "t_synthesis": round(_t_synthesis_ms, 1)}
            latency_metrics.record(_t_plan_ms, _t_execute_ms, _t_synthesis_ms)
            _record_turn(
                conversation, req.message, message, dataset_id, tool_calls, all_tool_results,
                tables=all_tables, charts=all_charts, citations=citations,
                groundedness_score=groundedness_score,
                groundedness_criteria=groundedness_criteria,
                groundedness_issues=groundedness_issues,
                judge_status=judge_status,
                planning_source=planning_source,
                synthesis_source=synthesis_source,
                latency_ms=_latency,
            )
            conversations.save(conversation)
        except Exception as e:
            logger.exception("Failed to record/save conversation turn: %s", e)

        try:
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
                judge_status=judge_status,
            )
            payload = json.dumps({"type": "done", "response": final.model_dump()})
        except (TypeError, ValueError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Response serialization failed: {exc}'})}\n\n"
            return
        except Exception as exc:
            logger.exception("Unexpected error building final response: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Internal error: {exc}'})}\n\n"
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
            citations=t.citations,
            tool_results=[ToolResult(**tr) for tr in t.tool_results],
            groundedness_score=t.groundedness_score,
            groundedness_criteria=t.groundedness_criteria,
            groundedness_issues=t.groundedness_issues,
            judge_status=cast(JudgeStatus, t.judge_status),
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
    citations=None,
    groundedness_score=None,
    groundedness_criteria=None,
    groundedness_issues=None,
    judge_status="rule_based",
    planning_source="rules",
    synthesis_source="rules",
    latency_ms: dict | None = None,
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
            citations=[c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in (citations or [])],
            tool_results=[tr.model_dump() for tr in tool_results],
            groundedness_score=groundedness_score,
            groundedness_criteria=groundedness_criteria or {},
            groundedness_issues=groundedness_issues or [],
            judge_status=judge_status,
            planning_source=planning_source,
            synthesis_source=synthesis_source,
            latency_ms=latency_ms or {},
        )
    )
    if dataset_id:
        conversation.last_dataset_id = dataset_id

    for result in tool_results:
        if result.name == "train_supervised_model" and result.ok and isinstance(result.result, dict):
            model_id = result.result.get("model_id")
            if model_id and model_id not in conversation.trained_model_ids:
                conversation.trained_model_ids.append(model_id)
