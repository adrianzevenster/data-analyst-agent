# Data Analyst Agent

Python service and Streamlit UI for uploading datasets, planning analytic tool calls, and returning grounded analysis summaries. The backend supports profiling, quality checks, SQL, clustering, anomaly detection, RAG-assisted planning, ML prediction evaluation, and training baseline supervised learning models.

## Architecture

- `app/main.py` wires the FastAPI service.
- `app/api/` exposes upload, dataset, health, and chat routes.
- `app/agent/planner.py` chooses analytics tools from user requests using rules, optional RAG, and optional OpenAI-compatible LLM planning.
- `app/agent/executor.py` loads a dataset and executes registered tools.
- `app/analytics/` contains analytic tools and the tool registry.
- `app/analytics/ml_eval/` evaluates prediction outputs for classification, regression/forecasting, scored predictions, and precomputed metric tables.
- `app/analytics/ml_train/` trains baseline supervised learning models (classification/regression) against a target column, evaluates them via `ml_eval`, and persists them to `data/models/` for reuse with `score_with_model`.
- `app/rag/` manages local embedding and FAISS retrieval over the analytics corpus.
- `app/ui/streamlit_app.py` provides the demo UI.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt -r requirements-ui.txt
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Run the UI:

```bash
API_URL=http://localhost:8080 streamlit run app/ui/streamlit_app.py
```

Or run both with Docker Compose:

```bash
docker compose up --build
```

## Optional LLM Configuration

The LLM layer expects an OpenAI-compatible `/chat/completions` endpoint.

```bash
LLM_ENABLED=true
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=Qwen/Qwen3-32B
```

If disabled, the planner falls back to rule-based tool selection.

LLM-proposed tool calls are schema-validated against the real dataset and repaired/dropped if they hallucinate columns (see `llm_notes` in the chat response). Call performance (latency, tokens, error rate) is tracked in-process and exposed at `GET /health/llm`. A sampled fraction of LLM-synthesized responses (`LLM_JUDGE_SAMPLE_RATE`, default `0.1`; production compose default `0.05`) is additionally scored for groundedness by an LLM-as-judge call. Aggregate scores plus skipped/error counters are at `GET /health/llm-judge` and the Streamlit sidebar.

## Tests

```bash
pytest
```

The default test suite focuses on ML evaluation, planner behavior, and dataset storage contracts, and does not require a live LLM server.

A separate opt-in golden-query eval checks the *live* LLM planner's tool-selection accuracy against a real LLM server (requires `LLM_ENABLED=true` and a reachable `LLM_BASE_URL`):

```bash
pytest -m llm_eval -v
```

This writes a per-case report to `data/eval/llm_planner_golden.json` and fails if aggregate accuracy drops below a floor - use it as a regression check when changing models or prompts.

A second opt-in eval checks RAG retrieval quality (recall@k / precision@k) against a small, hand-labeled fixture corpus (`tests/fixtures/rag_corpus/`). It only needs the local embedding model, not an LLM server, and never touches the real `data/indexes/`:

```bash
pytest -m rag_eval -v
```

This writes `data/eval/rag_retrieval_eval.json`, fails if recall@5 drops below a floor, and is served live at `GET /health/rag-eval` (also shown in the Streamlit sidebar) - run it after changing the embedding model, chunking, or `LLM_RAG_MIN_SCORE`.

## Data Hygiene

User uploads and derived indexes are local runtime artifacts. Keep `data/uploads/`, `data/indexes/`, and `data/corpus/` out of git unless a small fixture is intentionally added under `tests/fixtures/`.
