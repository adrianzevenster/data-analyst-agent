from __future__ import annotations

import logging

import numpy as np

from app.core.config import settings
from app.rag.embedder import LocalEmbedder
from app.rag.reranker import CrossEncoderReranker
from app.rag.store import FaissStore

logger = logging.getLogger(__name__)

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

    def _try_hyde_embedding(self, query: str) -> np.ndarray | None:
        """Return the embedding of a LLM-generated hypothetical answer, or None.

        HyDE (Hypothetical Document Embeddings): averaging the hypothetical-answer
        embedding with the raw query embedding bridges the vocabulary gap between
        how users phrase questions and how the corpus phrases answers.
        Only attempted when llm_hyde_enabled=True and the LLM is reachable.
        """
        if not settings.llm_hyde_enabled:
            return None
        try:
            from app.agent.llm import LLMReasoner
            reasoner = LLMReasoner()
            if not reasoner.enabled:
                return None
            passage = reasoner.hyde_passage(query)
            if not passage:
                return None
            vec = np.array(self.embedder.embed([passage])[0], dtype="float32")
            logger.debug("HyDE passage embedded (%d chars)", len(passage))
            return vec
        except Exception as exc:
            logger.debug("HyDE embedding failed, using plain query: %s", exc)
            return None

    def retrieve(self, query: str, top_k: int = 6, min_score: float | None = None) -> list[dict]:
        """Retrieve up to top_k chunks, dropping any below min_score.

        Pipeline:
        1. (Optional) HyDE: average query embedding with a hypothetical-answer
           embedding when llm_hyde_enabled=True — bridges vocabulary gaps.
        2. Hybrid BM25 + cosine search over a widened pool (top_k * 4).
        3. Cross-encoder rerank when enabled and model is available locally.
        4. Apply min_score filter; scores after reranking are CE logits
           (unbounded), so min_score is only applied to the hybrid scores
           from step 2 when reranking is active.
        """
        floor = settings.llm_rag_min_score if min_score is None else min_score
        qemb = np.array(self.embedder.embed([query])[0], dtype="float32")

        # HyDE: blend hypothetical-answer embedding with raw query embedding.
        hyde_emb = self._try_hyde_embedding(query)
        if hyde_emb is not None:
            combined = qemb + hyde_emb
            norm = float(np.linalg.norm(combined))
            if norm > 0:
                qemb = (combined / norm).astype("float32")

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
