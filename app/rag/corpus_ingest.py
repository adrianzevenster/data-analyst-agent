from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.ingestion.chunking import chunk_text
from app.rag.embedder import LocalEmbedder
from app.rag.store import FaissStore, StoredChunk
from app.ingestion.loaders import load_pdf_text


def iter_corpus_files(corpus_dir: str):
    exts = {".txt", ".md", ".pdf"}
    for p in Path(corpus_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ingest_corpus(corpus_dir: str | None = None, index_dir: str | None = None) -> dict:
    corpus_dir = corpus_dir or settings.corpus_dir
    index_dir = index_dir or settings.index_dir

    store = FaissStore(index_dir=index_dir)
    store.reset()

    embedder = LocalEmbedder()

    all_chunks: list[StoredChunk] = []
    all_texts: list[str] = []

    for p in iter_corpus_files(corpus_dir):
        source_id = str(p.relative_to(corpus_dir))
        if p.suffix.lower() == ".pdf":
            b = p.read_bytes()
            ing = load_pdf_text(b, p.name)
            text = ing.payload if ing.kind == "text" else ""
        else:
            text = read_text_file(str(p))

        for i, ch in enumerate(chunk_text(text)):
            all_chunks.append(StoredChunk(source_id=f"{source_id}#chunk={i}", text=ch))
            all_texts.append(ch)

    if all_texts:
        embs = np.array(embedder.embed(all_texts), dtype="float32")
        store.add(embs, all_chunks)

    return {"chunks_indexed": len(all_chunks), "index_dir": index_dir}
