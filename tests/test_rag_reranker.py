from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.rag.reranker import CrossEncoderReranker
from app.rag.store import StoredChunk


def _chunks(*texts: str) -> list[tuple[StoredChunk, float]]:
    return [(StoredChunk(source_id=str(i), text=t), 0.9 - i * 0.1) for i, t in enumerate(texts)]


def test_unavailable_when_model_not_found():
    with patch("sentence_transformers.CrossEncoder.__init__", side_effect=Exception("not cached")):
        reranker = CrossEncoderReranker("nonexistent/model")
        assert reranker.available is False


def test_rerank_passthrough_when_unavailable():
    with patch("sentence_transformers.CrossEncoder.__init__", side_effect=Exception("not cached")):
        reranker = CrossEncoderReranker("nonexistent/model")
    candidates = _chunks("a", "b", "c")
    result = reranker.rerank("query", candidates, top_k=2)
    assert result == candidates[:2]


def test_rerank_reorders_by_ce_score():
    fake_model = MagicMock()
    # CE scores: first chunk gets -1, second gets +5 → second should win
    fake_model.predict.return_value = np.array([-1.0, 5.0])

    reranker = CrossEncoderReranker("fake/model")
    reranker._model = fake_model
    reranker._available = True

    candidates = _chunks("low relevance", "high relevance")
    result = reranker.rerank("query", candidates, top_k=2)

    assert result[0][0].text == "high relevance"
    assert result[1][0].text == "low relevance"


def test_rerank_honours_top_k():
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])

    reranker = CrossEncoderReranker("fake/model")
    reranker._model = fake_model
    reranker._available = True

    candidates = _chunks("a", "b", "c", "d")
    result = reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2


def test_rerank_returns_empty_for_no_candidates():
    reranker = CrossEncoderReranker("fake/model")
    reranker._model = MagicMock()
    reranker._available = True
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


def test_rerank_falls_back_when_predict_raises():
    fake_model = MagicMock()
    fake_model.predict.side_effect = RuntimeError("GPU OOM")

    reranker = CrossEncoderReranker("fake/model")
    reranker._model = fake_model
    reranker._available = True

    candidates = _chunks("a", "b")
    result = reranker.rerank("query", candidates, top_k=2)

    # Fallback: original hybrid order
    assert result == candidates[:2]
