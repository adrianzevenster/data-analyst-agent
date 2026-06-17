from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from pydantic import BaseModel, ConfigDict


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class EmptyArgs(ToolArgs):
    pass


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[ToolArgs] = EmptyArgs

    def validate_args(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        return self.args_model.model_validate(arguments or {}).model_dump(exclude_none=True)


class AnalyticsToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def list(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": t.args_model.model_json_schema(),
            }
            for t in self.tools.values()
        ]
