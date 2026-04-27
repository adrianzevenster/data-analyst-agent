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
    message: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)