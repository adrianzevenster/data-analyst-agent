from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal


JudgeStatus = Literal["judged", "not_sampled", "rule_based", "llm_disabled", "failed"]


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int | None = None
    n_cols: int | None = None
    notes: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    dataset_id: str | None = None
    message: str
    top_k: int = 6
    conversation_id: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    result: Any = None
    error: str | None = None


class ChatResponse(BaseModel):
    dataset_id: str | None
    conversation_id: str
    message: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)

    llm_enabled: bool = False
    planning_source: Literal["llm", "rules"] = "rules"
    synthesis_source: Literal["llm", "rules"] = "rules"
    llm_error: str | None = None
    llm_notes: list[str] = Field(default_factory=list)
    groundedness_score: int | None = None
    groundedness_criteria: dict[str, int] = Field(default_factory=dict)
    groundedness_issues: list[str] = Field(default_factory=list)
    judge_status: JudgeStatus = "rule_based"


class TurnOut(BaseModel):
    role: str
    content: str
    dataset_id: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: float
    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    groundedness_score: int | None = None
    groundedness_criteria: dict[str, int] = Field(default_factory=dict)
    groundedness_issues: list[str] = Field(default_factory=list)
    judge_status: JudgeStatus = "rule_based"
    planning_source: str = "rules"
    synthesis_source: str = "rules"
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    turns: list[TurnOut] = Field(default_factory=list)


class LLMOperationStats(BaseModel):
    count: int
    errors: int
    avg_latency_ms: float


class LLMStatsResponse(BaseModel):
    window_size: int
    error_count: int
    error_rate: float
    avg_latency_ms: float
    total_tokens_sampled: int
    by_operation: dict[str, LLMOperationStats]


class JudgeStatsResponse(BaseModel):
    response_count: int
    eligible_count: int
    attempted_count: int
    sampled_count: int
    skipped_count: int
    skipped_sample_rate_count: int
    skipped_rule_based_count: int
    skipped_llm_disabled_count: int
    error_count: int
    avg_groundedness_score: float
    low_score_rate: float
    flagged_rate: float
    last_error: str | None = None


class RagEvalQueryResult(BaseModel):
    query: str
    expected_source: str
    hit: bool
    rank: int | None = None  # 1-based rank of first match; None if missed
    top_sources: list[str] = Field(default_factory=list)
    score: float | None = None  # similarity score of the top hit


class RagEvalResponse(BaseModel):
    available: bool
    n_queries: int = 0
    recall_at_1: float | None = None
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    mrr: float | None = None
    generated_at: float | None = None
    queries: list[RagEvalQueryResult] = Field(default_factory=list)


class RepairStatsResponse(BaseModel):
    repair_attempts: int
    total_problems: int
    total_fixed: int
    total_dropped: int
    fix_rate: float


class JudgeHistoryEntry(BaseModel):
    score: int
    issue_count: int
    synthesis_source: str
    timestamp: float


class JudgeHistoryResponse(BaseModel):
    entries: list[JudgeHistoryEntry]
    total: int


class PlannerFallbackResponse(BaseModel):
    total_fallbacks: int
    by_reason: dict[str, int]


class LatencyPhaseStats(BaseModel):
    avg_ms: float
    p50_ms: float
    p95_ms: float


class LatencyStatsResponse(BaseModel):
    n_turns: int
    phases: dict[str, LatencyPhaseStats]


class QualityTrendDay(BaseModel):
    day: str
    avg_score: float
    n: int
    min_score: int
    max_score: int


class QualityTrendResponse(BaseModel):
    days: int
    data: list[QualityTrendDay]


class EvalRunResult(BaseModel):
    run_id: str
    n_sampled: int
    n_judged: int
    n_failed: int
    avg_score: float | None = None


class EvalRunHistoryEntry(BaseModel):
    run_id: str
    n_sampled: int
    n_judged: int
    n_failed: int
    avg_score: float | None = None
    timestamp: float


class EvalRunHistoryResponse(BaseModel):
    entries: list[EvalRunHistoryEntry]


class ScoringModelLatency(BaseModel):
    n: int
    avg_ms: float
    p50_ms: float
    p95_ms: float


class ScoringLatencyResponse(BaseModel):
    n_models: int
    by_model: dict[str, ScoringModelLatency]
