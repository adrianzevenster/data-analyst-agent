from __future__ import annotations

import logging
import warnings

from app.rag.store import StoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for RAG retrieval.

    Uses a sentence-transformers CrossEncoder to jointly score (query, passage)
    pairs — far more accurate than bi-encoder cosine similarity alone, at the
    cost of O(candidates) forward passes (acceptable for small RAG corpora).

    Gracefully degrades: if the model isn't available locally, `available`
    returns False and `rerank()` passes candidates through unchanged.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._available: bool | None = None  # None = not yet probed

    @property
    def available(self) -> bool:
        if self._available is None:
            self._probe()
        return bool(self._available)

    def _probe(self) -> None:
        try:
            from sentence_transformers import CrossEncoder as CE

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
                try:
                    self._model = CE(self.model_name, local_files_only=True)
                except Exception:
                    self._model = CE(self.model_name)
            self._available = True
            logger.info("reranker loaded model=%s", self.model_name)
        except Exception as exc:
            self._available = False
            logger.info("reranker unavailable model=%s reason=%s", self.model_name, exc)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[StoredChunk, float]],
        top_k: int,
    ) -> list[tuple[StoredChunk, float]]:
        """Rerank `candidates` by cross-encoder score and return top_k.

        Falls back to original order when cross-encoder is unavailable.
        """
        if not candidates:
            return []

        if not self.available or self._model is None:
            return candidates[:top_k]

        pairs = [[query, chunk.text] for chunk, _ in candidates]
        try:
            scores: list[float] = self._model.predict(pairs).tolist()
        except Exception as exc:
            logger.warning("reranker.predict failed: %s — falling back to hybrid order", exc)
            return candidates[:top_k]

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: -x[1],
        )
        return [(chunk, ce_score) for (chunk, _), ce_score in reranked[:top_k]]
