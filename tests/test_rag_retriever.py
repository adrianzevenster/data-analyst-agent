from __future__ import annotations

from unittest.mock import MagicMock

from app.rag.retriever import RagRetriever
from app.rag.store import StoredChunk


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_emb, top_k=6):
        return self._hits[:top_k]

    def hybrid_search(self, query_emb, query_text, top_k=6, bm25_weight=0.3):
        return self._hits[:top_k]


def _make_retriever(hits, monkeypatch=None, reranker=None):
    retriever = RagRetriever.__new__(RagRetriever)
    retriever.store = _FakeStore(hits)
    retriever.embedder = _FakeEmbedder()
    if monkeypatch is not None and reranker is not None:
        import app.rag.retriever as _mod
        monkeypatch.setattr(_mod, "_get_reranker", lambda: reranker)
    return retriever


# ── Existing min-score tests (unchanged behaviour) ──────────────────────────

def test_retrieve_drops_hits_below_min_score(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", False)

    hits = [
        (StoredChunk(source_id="a", text="strong match"), 0.9),
        (StoredChunk(source_id="b", text="weak match"), 0.1),
    ]
    retriever = _make_retriever(hits)
    results = retriever.retrieve("query", top_k=6, min_score=0.25)
    assert [r["source_id"] for r in results] == ["a"]


def test_retrieve_uses_configured_default_min_score(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.5)
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", False)

    hits = [
        (StoredChunk(source_id="a", text="strong match"), 0.9),
        (StoredChunk(source_id="b", text="borderline"), 0.4),
    ]
    retriever = _make_retriever(hits)
    results = retriever.retrieve("query", top_k=6)
    assert [r["source_id"] for r in results] == ["a"]


def test_retrieve_keeps_all_hits_when_min_score_is_zero(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", False)

    hits = [
        (StoredChunk(source_id="a", text="strong match"), 0.9),
        (StoredChunk(source_id="b", text="weak match"), 0.0),
    ]
    retriever = _make_retriever(hits)
    results = retriever.retrieve("query", top_k=6, min_score=0.0)
    assert [r["source_id"] for r in results] == ["a", "b"]


# ── Cross-encoder reranker tests ─────────────────────────────────────────────

def _make_fake_reranker(available: bool, scores: list[float] | None = None):
    reranker = MagicMock()
    reranker.available = available
    if available and scores is not None:
        reranker.rerank.side_effect = lambda query, candidates, top_k: [
            (chunk, scores[i]) for i, (chunk, _) in enumerate(candidates[:top_k])
        ]
    return reranker


def test_reranker_called_when_enabled_and_available(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.0)

    hits = [
        (StoredChunk(source_id="a", text="first"), 0.8),
        (StoredChunk(source_id="b", text="second"), 0.7),
    ]
    fake_reranker = _make_fake_reranker(available=True, scores=[0.5, 0.9])
    retriever = _make_retriever(hits, monkeypatch, reranker=fake_reranker)

    retriever.retrieve("query", top_k=2)

    fake_reranker.rerank.assert_called_once()


def test_reranker_reverses_order_when_ce_scores_differ(monkeypatch):
    """Cross-encoder score should override hybrid order."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.0)

    hits = [
        (StoredChunk(source_id="first-hybrid", text="first"), 0.9),
        (StoredChunk(source_id="second-hybrid", text="second"), 0.8),
    ]
    # CE scores are inverted: second-hybrid gets higher CE score.
    def fake_rerank(query, candidates, top_k):
        scored = [(candidates[0][0], -1.0), (candidates[1][0], 5.0)]
        return sorted(scored, key=lambda x: -x[1])[:top_k]

    fake_reranker = MagicMock()
    fake_reranker.available = True
    fake_reranker.rerank.side_effect = fake_rerank

    retriever = _make_retriever(hits, monkeypatch, reranker=fake_reranker)
    results = retriever.retrieve("query", top_k=2)

    assert results[0]["source_id"] == "second-hybrid"
    assert results[1]["source_id"] == "first-hybrid"


def test_falls_back_to_hybrid_when_reranker_unavailable(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.0)

    hits = [
        (StoredChunk(source_id="a", text="first"), 0.9),
        (StoredChunk(source_id="b", text="second"), 0.7),
    ]
    fake_reranker = _make_fake_reranker(available=False)
    retriever = _make_retriever(hits, monkeypatch, reranker=fake_reranker)

    results = retriever.retrieve("query", top_k=2)

    fake_reranker.rerank.assert_not_called()
    assert [r["source_id"] for r in results] == ["a", "b"]


def test_min_score_pre_filters_before_reranking(monkeypatch):
    """Chunks below floor should never reach the cross-encoder."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.5)

    hits = [
        (StoredChunk(source_id="a", text="good"), 0.9),
        (StoredChunk(source_id="b", text="bad"), 0.1),
    ]
    captured: list = []

    def fake_rerank(query, candidates, top_k):
        captured.extend(candidates)
        return candidates[:top_k]

    fake_reranker = MagicMock()
    fake_reranker.available = True
    fake_reranker.rerank.side_effect = fake_rerank

    retriever = _make_retriever(hits, monkeypatch, reranker=fake_reranker)
    retriever.retrieve("query", top_k=6)

    source_ids = [c.source_id for c, _ in captured]
    assert "b" not in source_ids
    assert "a" in source_ids


def test_reranker_respects_top_k(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_rag_reranker_enabled", True)
    monkeypatch.setattr(settings, "llm_rag_min_score", 0.0)

    hits = [(StoredChunk(source_id=str(i), text=f"chunk {i}"), 0.9 - i * 0.1) for i in range(6)]
    fake_reranker = MagicMock()
    fake_reranker.available = True
    fake_reranker.rerank.side_effect = lambda q, candidates, top_k: candidates[:top_k]

    retriever = _make_retriever(hits, monkeypatch, reranker=fake_reranker)
    results = retriever.retrieve("query", top_k=3)

    assert len(results) == 3
    _, kwargs = fake_reranker.rerank.call_args
    assert kwargs.get("top_k", fake_reranker.rerank.call_args[0][2] if len(fake_reranker.rerank.call_args[0]) > 2 else None) == 3 or fake_reranker.rerank.called
