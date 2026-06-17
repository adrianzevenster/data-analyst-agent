"""Integration tests for the synchronous POST /chat endpoint."""
from __future__ import annotations

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def uploaded_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")
    df = pd.DataFrame({
        "region": ["east", "west", "north", "south"],
        "revenue": [100.0, 200.0, 150.0, 80.0],
        "units": [1, 2, 3, 1],
    })
    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        resp = client.post("/uploads", files={"file": ("sales.csv", f, "text/csv")})
    assert resp.status_code == 200
    return resp.json()["dataset_id"]


def test_chat_profile_returns_200(uploaded_dataset):
    resp = client.post("/chat", json={"dataset_id": uploaded_dataset, "message": "Profile this dataset"})
    assert resp.status_code == 200

    body = resp.json()
    assert "conversation_id" in body
    assert "message" in body
    assert isinstance(body["tool_results"], list)
    assert len(body["tool_results"]) > 0


def test_chat_empty_message_returns_400(uploaded_dataset):
    resp = client.post("/chat", json={"dataset_id": uploaded_dataset, "message": "   "})
    assert resp.status_code == 400


def test_chat_persists_conversation_id(uploaded_dataset):
    """Second turn should re-use conversation_id from first turn."""
    r1 = client.post("/chat", json={"dataset_id": uploaded_dataset, "message": "Profile this dataset"})
    assert r1.status_code == 200
    conv_id = r1.json()["conversation_id"]

    r2 = client.post("/chat", json={"dataset_id": uploaded_dataset, "message": "Show outliers", "conversation_id": conv_id})
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id


def test_chat_no_dataset_returns_200():
    resp = client.post("/chat", json={"message": "Hello"})
    assert resp.status_code == 200


def test_chat_tool_results_have_expected_shape(uploaded_dataset):
    resp = client.post("/chat", json={"dataset_id": uploaded_dataset, "message": "Profile this dataset"})
    assert resp.status_code == 200

    for tr in resp.json()["tool_results"]:
        assert "name" in tr
        assert "ok" in tr
