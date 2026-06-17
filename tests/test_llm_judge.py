from __future__ import annotations

import json

import pytest

from app.agent.llm import LLMReasoner, LLMUnavailable
from app.core.models import ToolResult


def _patch_chat(monkeypatch, reasoner: LLMReasoner, response: str) -> None:
    def fake_chat(messages, *, temperature=None, operation="chat"):
        return response

    monkeypatch.setattr(reasoner, "_chat", fake_chat)


def test_judge_groundedness_parses_score_and_issues(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(
        monkeypatch,
        reasoner,
        json.dumps({"groundedness_score": 4, "unsupported_claims": ["claims revenue grew 50%"]}),
    )

    verdict = reasoner.judge_groundedness(
        "Revenue grew 50% last quarter.",
        dataset_context={"column_profiles": []},
        tool_results=[ToolResult(name="profile_dataset", ok=True, result={})],
    )

    assert verdict == {"score": 4, "issues": ["claims revenue grew 50%"]}


def test_judge_groundedness_clamps_out_of_range_score(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(monkeypatch, reasoner, json.dumps({"groundedness_score": 9, "unsupported_claims": []}))

    verdict = reasoner.judge_groundedness("answer", dataset_context=None, tool_results=[])

    assert verdict["score"] == 5


def test_judge_groundedness_raises_on_invalid_json(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(monkeypatch, reasoner, "not json")

    with pytest.raises(LLMUnavailable):
        reasoner.judge_groundedness("answer", dataset_context=None, tool_results=[])
