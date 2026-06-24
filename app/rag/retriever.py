from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.rag.embedder import LocalEmbedder
from app.rag.reranker import CrossEncoderReranker
from app.rag.store import FaissStore

# Module-level singleton — shared across requests, loaded once.
_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(settings.llm_rag_reranker_model)
    return _reranker


class RagRetriever:
    def __init__(self, index_dir: str | None = None):
        self.store = FaissStore(index_dir=index_dir or str(settings.index_path))
        self.embedder = LocalEmbedder()

    def retrieve(self, query: str, top_k: int = 6, min_score: float | None = None) -> list[dict]:
        """Retrieve up to top_k chunks, dropping any below min_score.

        Pipeline:
        1. Hybrid BM25 + cosine search over a widened pool (top_k * 4).
        2. Cross-encoder rerank when enabled and model is available locally.
        3. Apply min_score filter; scores after reranking are CE logits
           (unbounded), so min_score is only applied to the hybrid scores
           from step 1 when reranking is active.
        """
        floor = settings.llm_rag_min_score if min_score is None else min_score
        qemb = np.array(self.embedder.embed([query])[0], dtype="float32")

        rerank_enabled = settings.llm_rag_reranker_enabled
        reranker = _get_reranker() if rerank_enabled else None

        if rerank_enabled and reranker is not None and reranker.available:
            # Widen candidate pool so the reranker has more to work with.
            pool_k = top_k * 4
            candidates = self.store.hybrid_search(qemb, query, top_k=pool_k)
            # Pre-filter below floor before sending to the (slower) cross-encoder.
            candidates = [(c, s) for c, s in candidates if s >= floor]
            reranked = reranker.rerank(query, candidates, top_k=top_k)
            # After reranking, scores are cross-encoder logits — don't apply
            # the cosine min_score; just return whatever the CE ranked highest.
            return [{"source_id": c.source_id, "text": c.text, "score": s} for c, s in reranked]
        else:
            hits = self.store.hybrid_search(qemb, query, top_k=top_k)
            return [{"source_id": h.source_id, "text": h.text, "score": s} for h, s in hits if s >= floor]
