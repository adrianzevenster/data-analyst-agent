from __future__ import annotations

import json

import pandas as pd

from app.agent.llm import LLMReasoner

DF = pd.DataFrame(
    {
        "region": ["east", "west", "east", "west"],
        "revenue": [100.0, 200.0, 150.0, 50.0],
    }
)


def _responses(*payloads: dict) -> list[str]:
    return [json.dumps(p) for p in payloads]


def _patch_chat(monkeypatch, reasoner: LLMReasoner, responses: list[str]) -> None:
    calls = iter(responses)

    def fake_chat(messages, *, temperature=None, operation="chat"):
        return next(calls)

    monkeypatch.setattr(reasoner, "_chat", fake_chat)


def test_plan_passes_through_valid_tool_calls(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(
        monkeypatch,
        reasoner,
        _responses({"tool_calls": [{"name": "profile_dataset", "arguments": {"sample": 100}}]}),
    )

    calls, notes = reasoner.plan("profile this", dataset_id="d1", df=DF, citations=[])

    assert [c.name for c in calls] == ["profile_dataset"]
    assert notes == []


def test_plan_drops_hallucinated_column_when_repair_cannot_fix_it(monkeypatch):
    reasoner = LLMReasoner()
    bad_call = {
        "tool_calls": [
            {"name": "histogram_spec", "arguments": {"column": "does_not_exist"}}
        ]
    }
    _patch_chat(monkeypatch, reasoner, _responses(bad_call, bad_call))

    calls, notes = reasoner.plan("histogram of nothing", dataset_id="d1", df=DF, citations=[])

    assert calls == []
    assert len(notes) == 1
    assert "histogram_spec" in notes[0]
    assert "does_not_exist" in notes[0]


def test_plan_repairs_hallucinated_column_using_real_schema(monkeypatch):
    reasoner = LLMReasoner()
    bad_call = {
        "tool_calls": [
            {"name": "histogram_spec", "arguments": {"column": "does_not_exist"}}
        ]
    }
    fixed_call = {
        "tool_calls": [
            {"name": "histogram_spec", "arguments": {"column": "revenue"}}
        ]
    }
    _patch_chat(monkeypatch, reasoner, _responses(bad_call, fixed_call))

    calls, notes = reasoner.plan("histogram of revenue", dataset_id="d1", df=DF, citations=[])

    assert [c.name for c in calls] == ["histogram_spec"]
    assert calls[0].arguments["column"] == "revenue"
    assert notes == []


def test_plan_drops_only_invalid_calls_and_keeps_valid_ones(monkeypatch):
    reasoner = LLMReasoner()
    mixed_call = {
        "tool_calls": [
            {"name": "profile_dataset", "arguments": {"sample": 100}},
            {"name": "histogram_spec", "arguments": {"column": "does_not_exist"}},
        ]
    }
    repair_drops_bad_call = {"tool_calls": []}
    _patch_chat(monkeypatch, reasoner, _responses(mixed_call, repair_drops_bad_call))

    calls, notes = reasoner.plan("profile and histogram", dataset_id="d1", df=DF, citations=[])

    assert [c.name for c in calls] == ["profile_dataset"]
    assert len(notes) == 1
    assert "histogram_spec" in notes[0]


def test_plan_skips_validation_when_no_dataframe_available(monkeypatch):
    reasoner = LLMReasoner()
    _patch_chat(
        monkeypatch,
        reasoner,
        _responses({"tool_calls": [{"name": "histogram_spec", "arguments": {"column": "anything"}}]}),
    )

    calls, notes = reasoner.plan("histogram", dataset_id=None, df=None, citations=[])

    assert [c.name for c in calls] == ["histogram_spec"]
    assert notes == []
