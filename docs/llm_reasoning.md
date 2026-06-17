# LLM Reasoning Layer

The API can optionally use a local/open-weight LLM through an OpenAI-compatible
`/v1/chat/completions` endpoint. When enabled, the LLM performs two bounded jobs:

1. Plan analytics tool calls from the user request, dataset schema, sample rows,
   and retrieved corpus context.
2. Synthesize a concise analyst response from executed tool results and a
   deterministic dataset analysis context.

The existing deterministic planner remains the fallback when the LLM is disabled,
unavailable, or returns unusable tool JSON.

The LLM does not compute over the full raw file directly. The API first builds a
bounded, deterministic context containing schema, missingness, numeric summaries,
categorical top values, sample rows, and strongest numeric correlations. This
keeps the model grounded while still allowing dataset-specific interpretation.

Retrieved corpus chunks are passed to the LLM as bounded `rag_context` entries
with source IDs, retrieval scores, and excerpts. The LLM uses this context for
domain guidance, while dataset facts and tool outputs remain the source of truth
for claims about the uploaded data.

## Recommended Open-Weight Models

Use an OpenAI-compatible serving layer such as vLLM, SGLang, Ollama, or LM Studio.

| Use case | Recommended model | Why |
| --- | --- | --- |
| Best practical default | `Qwen/Qwen3-32B` | Strong reasoning, coding, and instruction following at a size that is realistic for a high-memory single GPU or quantized local deployment. |
| Highest reasoning quality | `Qwen/Qwen3-235B-A22B` or a DeepSeek-R1 class model | Better deep reasoning for complex analytical planning, but requires multi-GPU serving or aggressive quantization. |
| Tool-call reliability first | Mistral Small 3.2 | Official function-calling support and good mid-size agent behavior. |
| Lightweight local testing | Qwen3 8B/14B or a distilled DeepSeek-R1 variant | Lower cost and easier local iteration, with less reliable planning on ambiguous data questions. |

For this application, `Qwen/Qwen3-32B` is the default because the agent needs a
balance of reasoning quality, JSON discipline, and practical deployability.

## Configuration

Set these variables in `.env` or the API container environment:

```bash
LLM_ENABLED=true
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=
LLM_MODEL=Qwen/Qwen3-32B
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=120
LLM_MAX_TOOL_CALLS=4
LLM_ANALYSIS_SAMPLE_ROWS=5000
LLM_ANALYSIS_PREVIEW_ROWS=5
LLM_ANALYSIS_MAX_COLUMNS=80
LLM_RAG_MAX_CHUNKS=6
LLM_RAG_MAX_CHARS_PER_CHUNK=1200
```

When running through Docker Compose, `LLM_BASE_URL` defaults to
`http://host.docker.internal:8000/v1` so the API container can call a model server
running on the host.

## Example vLLM Server

```bash
vllm serve Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768
```

For Qwen3 thinking models, keep planning prompts strict and low temperature.
The agent validates returned tool names against the local registry before
execution.
