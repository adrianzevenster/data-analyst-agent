"""Integration tests for the SSE streaming chat endpoint."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _parse_sse_events(raw: str) -> list[dict]:
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


@pytest.fixture
def uploaded_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    df = pd.DataFrame({
        "region": ["east", "west", "north", "south"],
        "revenue": [100.0, 200.0, 150.0, 80.0],
        "units": [1, 2, 3, 1],
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        resp = client.post("/uploads", files={"file": ("test.csv", f, "text/csv")})
    assert resp.status_code == 200
    return resp.json()["dataset_id"]


def test_stream_returns_plan_toolresult_done_events(uploaded_dataset):
    with client.stream(
        "POST",
        "/chat/stream",
        json={"dataset_id": uploaded_dataset, "message": "Profile this dataset"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        raw = r.read().decode()

    events = _parse_sse_events(raw)
    types = [e["type"] for e in events]

    assert "plan" in types, f"Missing 'plan' event. Got: {types}"
    assert "tool_result" in types, f"Missing 'tool_result' event. Got: {types}"
    assert "done" in types, f"Missing 'done' event. Got: {types}"

    done_event = next(e for e in events if e["type"] == "done")
    response = done_event["response"]
    assert "conversation_id" in response
    assert "message" in response
    assert isinstance(response["tool_results"], list)
    assert len(response["tool_results"]) > 0


def test_stream_empty_message_returns_400(uploaded_dataset):
    resp = client.post("/chat/stream", json={"dataset_id": uploaded_dataset, "message": "  "})
    assert resp.status_code == 400


def test_stream_conversation_id_consistent_across_events(uploaded_dataset):
    with client.stream(
        "POST",
        "/chat/stream",
        json={"dataset_id": uploaded_dataset, "message": "Find outliers"},
    ) as r:
        raw = r.read().decode()

    events = _parse_sse_events(raw)

    plan_event = next((e for e in events if e["type"] == "plan"), None)
    done_event = next((e for e in events if e["type"] == "done"), None)

    assert plan_event is not None
    assert done_event is not None
    assert plan_event["conversation_id"] == done_event["response"]["conversation_id"]


def test_stream_tool_results_match_planned_tools(uploaded_dataset):
    with client.stream(
        "POST",
        "/chat/stream",
        json={"dataset_id": uploaded_dataset, "message": "Profile this dataset"},
    ) as r:
        raw = r.read().decode()

    events = _parse_sse_events(raw)

    plan_event = next((e for e in events if e["type"] == "plan"), None)
    tool_result_events = [e for e in events if e["type"] == "tool_result"]

    assert plan_event is not None
    planned_names = {tc["name"] for tc in plan_event.get("tool_calls", [])}
    result_names = {e["name"] for e in tool_result_events}

    assert planned_names == result_names, (
        f"Planned tools {planned_names} don't match result tools {result_names}"
    )


def test_stream_no_dataset_still_returns_done(monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "Hello"},
    ) as r:
        raw = r.read().decode()

    events = _parse_sse_events(raw)
    types = [e["type"] for e in events]
    assert "done" in types
