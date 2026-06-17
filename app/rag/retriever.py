from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.rag.embedder import LocalEmbedder
from app.rag.store import FaissStore


class RagRetriever:
    def __init__(self, index_dir: str | None = None):
        self.store = FaissStore(index_dir=index_dir or str(settings.index_path))
        self.embedder = LocalEmbedder()

    def retrieve(self, query: str, top_k: int = 6, min_score: float | None = None) -> list[dict]:
        """Retrieve up to top_k chunks, dropping any below min_score.

        FAISS returns cosine similarity (embeddings are normalized), so
        higher is better; weak matches add noise to the LLM prompt rather
        than useful grounding, so they're filtered here rather than left for
        every caller to filter individually.
        """
        floor = settings.llm_rag_min_score if min_score is None else min_score
        qemb = np.array(self.embedder.embed([query])[0], dtype="float32")
        hits = self.store.search(qemb, top_k=top_k)
        return [{"source_id": h.source_id, "text": h.text, "score": s} for h, s in hits if s >= floor]
