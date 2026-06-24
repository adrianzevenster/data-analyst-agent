from __future__ import annotations

import os
import warnings
from typing import List


class LocalEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Add it to requirements-api.txt."
            ) from e

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

            # First attempt: force offline so the HF Hub client reads the local
            # snapshot cache directly without making any network request.
            # This works even when HF is unreachable, as long as the model was
            # previously cached (e.g. via `huggingface-cli download`).
            saved = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                self._model = SentenceTransformer(self.model_name)
                return
            except Exception:
                pass
            finally:
                if saved is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = saved

            # Second attempt: allow hub download (requires internet).
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
