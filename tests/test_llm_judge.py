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
        json.dumps({
            "groundedness": 4,
            "accuracy": 4,
            "completeness": 4,
            "unsupported_claims": ["claims revenue grew 50%"],
        }),
    )

    verdict = reasoner.judge_groundedness(
        "Revenue grew 50% last quarter.",
        dataset_context={"column_profiles": []},
        tool_results=[ToolResult(name="profile_dataset", ok=True, result={})],
    )

    assert verdict["score"] == 4
    assert verdict["issues"] == ["claims revenue grew 50%"]
    assert "criteria" in verdict


def test_judge_groundedness_clamps_out_of_range_score(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(monkeypatch, reasoner, json.dumps({
        "groundedness": 9,
        "accuracy": 9,
        "completeness": 9,
        "unsupported_claims": [],
    }))

    verdict = reasoner.judge_groundedness("answer", dataset_context=None, tool_results=[])

    assert verdict["score"] == 5


def test_judge_groundedness_raises_on_invalid_json(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(monkeypatch, reasoner, "not json")

    with pytest.raises(LLMUnavailable):
        reasoner.judge_groundedness("answer", dataset_context=None, tool_results=[])


def test_first_eligible_response_is_sampled(monkeypatch):
    from app.api import routes_chat

    class FakeJudgeMetrics:
        def snapshot(self):
            return {"attempted_count": 0, "sampled_count": 0}

    monkeypatch.setattr(routes_chat, "judge_metrics", FakeJudgeMetrics())
    monkeypatch.setattr(routes_chat.settings, "llm_judge_sample_rate", 0.1)
    monkeypatch.setattr(
        routes_chat.random,
        "random",
        lambda: pytest.fail("first eligible response should not rely on random sampling"),
    )

    assert routes_chat._should_sample_judge() is True


def test_zero_judge_sample_rate_disables_sampling(monkeypatch):
    from app.api import routes_chat

    monkeypatch.setattr(routes_chat.settings, "llm_judge_sample_rate", 0.0)

    assert routes_chat._should_sample_judge() is False
