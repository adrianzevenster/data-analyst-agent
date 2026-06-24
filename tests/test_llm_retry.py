from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.agent.llm import LLMReasoner, LLMUnavailable

_GOOD_BODY = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"total_tokens": 10},
}
_EMPTY_BODY = {
    "choices": [{"message": {"content": ""}}],
    "usage": {"total_tokens": 3},
}


def _ok_resp(body: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = body if body is not None else _GOOD_BODY
    return resp


def _err_resp(status_code: int) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    http_err = requests.exceptions.HTTPError(response=resp)
    resp.raise_for_status.side_effect = http_err
    resp.json.return_value = {}
    return resp


@pytest.fixture()
def reasoner(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "llm_base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(settings, "llm_model", "test-model")
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    return LLMReasoner()


def test_success_on_first_attempt(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", return_value=_ok_resp()) as mock_post:
        result = reasoner._chat([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert mock_post.call_count == 1


def test_retries_on_5xx_then_succeeds(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", side_effect=[_err_resp(503), _ok_resp()]) as mock_post:
        result = reasoner._chat([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert mock_post.call_count == 2


def test_no_retry_on_4xx(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", return_value=_err_resp(400)) as mock_post:
        with pytest.raises(LLMUnavailable):
            reasoner._chat([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 1


def test_no_retry_on_401(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", return_value=_err_resp(401)) as mock_post:
        with pytest.raises(LLMUnavailable):
            reasoner._chat([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 1


def test_exhausted_retries_raises_llm_unavailable(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    # llm_max_retries=2 → 3 total attempts
    with patch("requests.post", return_value=_err_resp(503)) as mock_post:
        with pytest.raises(LLMUnavailable):
            reasoner._chat([{"role": "user", "content": "hi"}])
    assert mock_post.call_count == 3


def test_retries_on_empty_content_then_succeeds(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", side_effect=[_ok_resp(_EMPTY_BODY), _ok_resp()]) as mock_post:
        result = reasoner._chat([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert mock_post.call_count == 2


def test_retry_uses_exponential_backoff(reasoner, monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))
    with patch("requests.post", return_value=_err_resp(503)):
        with pytest.raises(LLMUnavailable):
            reasoner._chat([{"role": "user", "content": "hi"}])
    # 3 attempts → 2 sleep calls: delay=1 then delay=2
    assert sleep_calls == [1, 2]


def test_network_error_retries(reasoner, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    conn_err = requests.exceptions.ConnectionError("refused")
    with patch("requests.post", side_effect=[conn_err, _ok_resp()]) as mock_post:
        result = reasoner._chat([{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert mock_post.call_count == 2
