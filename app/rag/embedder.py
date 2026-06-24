from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List


def _resolve_local_model_path(model_name: str) -> str | None:
    """Return the local snapshot directory for a cached HF Hub model, or None.

    Reads the HF Hub on-disk cache layout directly — no network call.
    Works even when HF_HUB_OFFLINE is not set and the hub is unreachable.
    Checks all standard cache locations so it works regardless of which user
    or working directory the server process was started under.
    """
    slug = "models--" + model_name.replace("/", "--")

    # Candidate cache roots, in priority order
    candidates: list[Path] = []

    # 1. Explicitly set env vars (highest priority)
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        val = os.environ.get(env)
        if val:
            p = Path(val)
            candidates.append(p if env == "HUGGINGFACE_HUB_CACHE" else p / "hub")

    # 2. Standard default under the current user's home
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    # 3. /root and /home/* for deployments where HOME may differ
    for base in [Path("/root"), *Path("/home").glob("*")]:
        p = base / ".cache" / "huggingface" / "hub"
        if p not in candidates:
            candidates.append(p)

    for hub_dir in candidates:
        model_dir = hub_dir / slug
        refs_main = model_dir / "refs" / "main"
        if not refs_main.exists():
            continue
        snapshot_hash = refs_main.read_text().strip()
        snapshot_dir = model_dir / "snapshots" / snapshot_hash
        if snapshot_dir.exists() and (snapshot_dir / "config.json").exists():
            return str(snapshot_dir)

    return None


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

            # First attempt: resolve the snapshot directory directly from the
            # on-disk HF cache, bypassing any network or hub-client check.
            local_path = _resolve_local_model_path(self.model_name)
            if local_path:
                try:
                    self._model = SentenceTransformer(local_path)
                    return
                except Exception:
                    pass  # corrupted cache — fall through to hub download

            # Second attempt: hub download (requires internet access).
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
