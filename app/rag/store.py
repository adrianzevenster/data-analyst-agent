from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np


@dataclass
class StoredChunk:
    source_id: str
    text: str


class FaissStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "chunks.jsonl")

        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.chunks: list[StoredChunk] = []

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d
            self.chunks = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self.chunks.append(StoredChunk(**d))

    def reset(self) -> None:
        self.index = None
        self.dim = None
        self.chunks = []
        for p in [self.index_path, self.meta_path]:
            if os.path.exists(p):
                os.remove(p)

    def add(self, embeddings: np.ndarray, chunks: list[StoredChunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [n, d]")
        n, d = embeddings.shape

        if self.index is None:
            self.dim = d
            self.index = faiss.IndexFlatIP(d)  # cosine if embeddings are normalized
        elif d != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {d}")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    def search(self, query_emb: np.ndarray, top_k: int = 6) -> list[tuple[StoredChunk, float]]:
        if self.index is None or not self.chunks:
            return []
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        scores, idx = self.index.search(query_emb.astype("float32"), top_k)
        out: list[tuple[StoredChunk, float]] = []
        for i, s in zip(idx[0], scores[0]):
            if i < 0:
                continue
            out.append((self.chunks[int(i)], float(s)))
        return out
