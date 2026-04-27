from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import pandas as pd

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]


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
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]
