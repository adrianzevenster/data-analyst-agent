"""Recall@k / precision@k eval for RAG retrieval against a small, hand-labeled
fixture corpus (tests/fixtures/rag_corpus/).

Each fixture document is a short, single-topic analytics guidance note that
fits in one chunk (verified by chunk count == 1 per file), so "is the right
chunk retrieved" reduces cleanly to "is the right document retrieved" - no
ambiguity from a document being split across multiple chunks.

This indexes the fixture corpus into a throwaway FAISS index (never touches
the real data/indexes/ used by the running app) and runs a real embedding
model, so it's a bit slower than a pure unit test, but needs no LLM server -
gated behind the `rag_eval` marker mainly to keep the default suite from
depending on the embedding model being downloaded/cached.

Run with: pytest -m rag_eval -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import settings
from app.rag.corpus_ingest import ingest_corpus
from app.rag.retriever import RagRetriever

pytestmark = pytest.mark.rag_eval

FIXTURE_CORPUS_DIR = Path(__file__).parent / "fixtures" / "rag_corpus"
K_VALUES = [1, 3, 5]
MIN_RECALL_AT_5 = 0.7

# (query, expected relevant source files)
GOLDEN_QUERIES: list[tuple[str, set[str]]] = [
    ("How should I find unusual or anomalous rows in this data?", {"outlier_detection.md"}),
    ("What's the best way to detect outliers across many numeric columns?", {"outlier_detection.md"}),
    ("This dataset has a lot of null values, what should I do?", {"missing_data.md"}),
    ("How do I decide whether to drop a column with missing data?", {"missing_data.md"}),
    ("What correlates with revenue in this dataset?", {"correlation_analysis.md"}),
    ("How do I check association between a category and a numeric column?", {"correlation_analysis.md"}),
    ("Is this metric trending up or down over time?", {"trend_analysis.md"}),
    ("How should I pick the resampling frequency for a time series?", {"trend_analysis.md"}),
    ("How many clusters should I use to segment these customers?", {"clustering.md"}),
    ("What does KMeans assume about the shape of clusters?", {"clustering.md"}),
    ("How do I know if I should evaluate this as classification or regression?", {"model_evaluation.md"}),
    ("Why would WMAPE be better than MAPE for this forecast?", {"model_evaluation.md"}),
]


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("rag_eval_index"))
    stats = ingest_corpus(corpus_dir=str(FIXTURE_CORPUS_DIR), index_dir=index_dir)

    # Each fixture doc is short enough to be exactly one chunk; if that ever
    # stops being true (someone pads a fixture file), the file-level
    # expected-relevance labels below stop being meaningful, so fail loudly
    # instead of silently producing a misleading recall/precision number.
    n_fixture_files = len(list(FIXTURE_CORPUS_DIR.glob("*.md")))
    assert stats["chunks_indexed"] == n_fixture_files, (
        "Expected exactly one chunk per fixture file; a fixture doc grew "
        "past the chunk size, which invalidates the file-level labels."
    )

    return RagRetriever(index_dir=index_dir)


def _source_file(source_id: str) -> str:
    return source_id.split("#chunk=")[0]


def test_rag_retrieval_recall_and_precision_at_k(retriever):
    per_query_results = []

    for query, expected_files in GOLDEN_QUERIES:
        # min_score=0.0: this eval measures the embedding/retrieval ranking
        # itself, independent of the score-floor cutoff (which is evaluated
        # separately and could otherwise mask a ranking regression by just
        # filtering everything out).
        hits = retriever.retrieve(query, top_k=max(K_VALUES), min_score=0.0)
        retrieved_files_in_order = [_source_file(h["source_id"]) for h in hits]

        per_k = {}
        for k in K_VALUES:
            top_k_files = set(retrieved_files_in_order[:k])
            hit_files = top_k_files & expected_files
            per_k[k] = {
                "recall": len(hit_files) / len(expected_files) if expected_files else None,
                "precision": len(hit_files) / k,
            }

        per_query_results.append(
            {
                "query": query,
                "expected_files": sorted(expected_files),
                "retrieved_files": retrieved_files_in_order,
                "per_k": per_k,
            }
        )

    aggregate = {
        k: {
            "recall_at_k": round(
                sum(r["per_k"][k]["recall"] for r in per_query_results) / len(per_query_results), 4
            ),
            "precision_at_k": round(
                sum(r["per_k"][k]["precision"] for r in per_query_results) / len(per_query_results), 4
            ),
        }
        for k in K_VALUES
    }

    report = {
        "n_queries": len(per_query_results),
        "aggregate": aggregate,
        "min_recall_at_5": MIN_RECALL_AT_5,
        "per_query": per_query_results,
    }
    settings.eval_path.mkdir(parents=True, exist_ok=True)
    (settings.eval_path / "rag_retrieval_eval.json").write_text(json.dumps(report, indent=2, default=str))

    recall_at_5 = aggregate[5]["recall_at_k"]
    failures = [r for r in per_query_results if r["per_k"][5]["recall"] < 1.0]
    assert recall_at_5 >= MIN_RECALL_AT_5, (
        f"RAG recall@5 {recall_at_5:.2f} fell below floor {MIN_RECALL_AT_5}. "
        f"Failing queries: {json.dumps(failures, indent=2, default=str)}"
    )
