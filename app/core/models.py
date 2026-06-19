from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal


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


class TurnOut(BaseModel):
    role: str
    content: str
    dataset_id: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: float
    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    groundedness_score: int | None = None
    groundedness_criteria: dict[str, int] = Field(default_factory=dict)
    groundedness_issues: list[str] = Field(default_factory=list)
    planning_source: str = "rules"
    synthesis_source: str = "rules"


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
    sampled_count: int
    avg_groundedness_score: float
    low_score_rate: float
    flagged_rate: float


class RagEvalKStats(BaseModel):
    recall_at_k: float
    precision_at_k: float


class RagEvalResponse(BaseModel):
    available: bool
    n_queries: int = 0
    aggregate: dict[str, RagEvalKStats] = Field(default_factory=dict)
    min_recall_at_5: float | None = None


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