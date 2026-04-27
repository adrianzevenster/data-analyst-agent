from __future__ import annotations

from app.core.config import settings
from app.rag.embedder import LocalEmbedder
from app.rag.store import FaissStore


class RagRetriever:
    def __init__(self, index_dir: str | None = None):
        self.store = FaissStore(index_dir=index_dir or str(settings.index_path))
        self.embedder = LocalEmbedder()

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        qemb = self.embedder.embed([query])[0]
        hits = self.store.search(qemb, top_k=top_k)
        return [{"source_id": h.source_id, "text": h.text, "score": s} for h, s in hits]
