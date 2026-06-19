from __future__ import annotations

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


def _make_retriever(hits):
    retriever = RagRetriever.__new__(RagRetriever)
    retriever.store = _FakeStore(hits)
    retriever.embedder = _FakeEmbedder()
    return retriever


def test_retrieve_drops_hits_below_min_score():
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
    hits = [
        (StoredChunk(source_id="a", text="strong match"), 0.9),
        (StoredChunk(source_id="b", text="borderline"), 0.4),
    ]
    retriever = _make_retriever(hits)

    results = retriever.retrieve("query", top_k=6)

    assert [r["source_id"] for r in results] == ["a"]


def test_retrieve_keeps_all_hits_when_min_score_is_zero():
    hits = [
        (StoredChunk(source_id="a", text="strong match"), 0.9),
        (StoredChunk(source_id="b", text="weak match"), 0.0),
    ]
    retriever = _make_retriever(hits)

    results = retriever.retrieve("query", top_k=6, min_score=0.0)

    assert [r["source_id"] for r in results] == ["a", "b"]
