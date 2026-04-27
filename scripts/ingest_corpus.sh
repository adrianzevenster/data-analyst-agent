#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"
python -c "from app.rag.corpus_ingest import ingest_corpus; print(ingest_corpus())"
