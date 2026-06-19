from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class StoredChunk:
    source_id: str
    text: str


def _tokenize(text: str) -> list[str]:
    """Split into lowercase alphanumeric tokens, splitting on punctuation and underscores."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """BM25 keyword retrieval (Robertson & Zaragoza, 2009) — zero external dependencies.

    Rebuilt from chunk text on every process start (no separate persistence needed
    since the RAG corpus is small and load is fast).
    """

    k1: float = 1.5
    b: float = 0.75

    def __init__(self) -> None:
        self._docs: list[list[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0

    def add(self, texts: list[str]) -> None:
        tokenized = [_tokenize(t) for t in texts]
        self._docs.extend(tokenized)
        for tokens in tokenized:
            for term in set(tokens):
                self._df[term] += 1
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)

    def reset(self) -> None:
        self._docs.clear()
        self._df.clear()
        self._avgdl = 0.0

    def scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (doc_index, bm25_score) pairs, sorted descending, non-zero only."""
        terms = _tokenize(query)
        N = len(self._docs)
        if N == 0 or not terms:
            return []

        raw: list[float] = [0.0] * N
        for term in terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc in enumerate(self._docs):
                freq = doc.count(term)
                if freq == 0:
                    continue
                dl = len(doc)
                tf = freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                )
                raw[i] += idf * tf

        ranked = sorted(
            ((i, s) for i, s in enumerate(raw) if s > 0),
            key=lambda x: -x[1],
        )
        return ranked[:top_k]


class FaissStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "chunks.jsonl")

        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.chunks: list[StoredChunk] = []
        self.bm25 = BM25Index()

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
            # Rebuild BM25 from persisted chunks (fast; corpus is small)
            self.bm25.add([c.text for c in self.chunks])

    def reset(self) -> None:
        self.index = None
        self.dim = None
        self.chunks = []
        self.bm25.reset()
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
        self.bm25.add([c.text for c in chunks])

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    def search(self, query_emb: np.ndarray, top_k: int = 6) -> list[tuple[StoredChunk, float]]:
        """Pure semantic search (cosine similarity). Kept for backwards compatibility."""
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

    def hybrid_search(
        self,
        query_emb: np.ndarray,
        query_text: str,
        top_k: int = 6,
        bm25_weight: float = 0.3,
    ) -> list[tuple[StoredChunk, float]]:
        """BM25 + cosine hybrid: widens candidate pool, re-ranks by linear combination.

        bm25_weight=0.3 gives priority to semantic similarity while letting
        exact keyword matches (column names, metric abbreviations like WMAPE)
        boost relevant chunks that cosine alone would miss.
        """
        if self.index is None or not self.chunks:
            return []

        pool = min(top_k * 3, len(self.chunks))

        # Semantic candidates
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        faiss_scores, faiss_idx = self.index.search(query_emb.astype("float32"), pool)
        cosine_map: dict[int, float] = {
            int(i): float(s)
            for i, s in zip(faiss_idx[0], faiss_scores[0])
            if i >= 0
        }

        # Keyword candidates
        bm25_hits = self.bm25.scores(query_text, top_k=pool)
        bm25_map: dict[int, float] = {i: s for i, s in bm25_hits}

        # Union, normalize BM25 over the candidate pool, then combine
        candidate_ids = set(cosine_map) | set(bm25_map)
        bm25_max = max(bm25_map.values(), default=1.0)

        ranked: list[tuple[int, float]] = []
        for idx in candidate_ids:
            cosine = cosine_map.get(idx, 0.0)
            bm25_norm = bm25_map.get(idx, 0.0) / max(bm25_max, 1e-9)
            hybrid = (1.0 - bm25_weight) * cosine + bm25_weight * bm25_norm
            ranked.append((idx, hybrid))

        ranked.sort(key=lambda x: -x[1])
        return [
            (self.chunks[i], s)
            for i, s in ranked[:top_k]
            if 0 <= i < len(self.chunks)
        ]
