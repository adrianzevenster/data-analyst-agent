from __future__ import annotations

import os
from typing import List

class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed in the API container. "
                "Add it to requirements-api.txt OR switch to an offline embedder."
            ) from e

        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            # Model not in local cache — allow hub download as fallback
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


