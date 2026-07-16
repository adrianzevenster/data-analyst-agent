from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Data Analyst Agent"
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    data_dir: str = "./data"

    tesseract_cmd: str | None = None

    llm_enabled: bool = False
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen/Qwen3-32B"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 120
    llm_max_tool_calls: int = 4
    llm_max_repair_attempts: int = 1
    llm_max_retries: int = 2
    llm_analysis_sample_rows: int = 5000
    llm_analysis_preview_rows: int = 5
    llm_analysis_max_columns: int = 80
    llm_analysis_max_correlations: int = 20
    llm_analysis_max_categorical_associations: int = 10
    llm_rag_max_chunks: int = 6
    llm_rag_max_chars_per_chunk: int = 1200
    llm_rag_min_score: float = 0.25
    llm_rag_reranker_enabled: bool = True
    llm_rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # HyDE: generate a hypothetical answer, embed it, average with query embedding.
    # Improves recall when user phrasing differs from corpus language.
    # Requires llm_enabled=True. Each retrieval call costs one LLM round-trip.
    llm_hyde_enabled: bool = False
    llm_judge_sample_rate: float = 1.0
    llm_json_mode: bool = False  # set True when the inference server supports response_format
    # Always use json_object for plan/repair (set False only for backends that reject response_format)
    llm_json_plan: bool = True
    # When True, send json_schema with a tool-name enum for plan calls (requires vllm / llama.cpp ≥0.2)
    llm_json_schema_plan: bool = False

    model_registry_max_per_target: int = 10  # evict oldest when exceeded per (dataset_id, target_col)

    # When True, high-drift scoring events automatically enqueue a background retrain job
    auto_retrain_on_high_drift: bool = False
    auto_retrain_model_type: str = "auto"

    # After executing the first set of tools, the LLM gets one more planning step
    # to add follow-up tools (e.g. auto-explain after training). Set False to disable.
    llm_agentic_continuation: bool = True
    llm_max_agentic_steps: int = 2  # total planning rounds per request

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def upload_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def corpus_path(self) -> Path:
        return self.data_path / "corpus"

    @property
    def index_path(self) -> Path:
        return self.data_path / "indexes"

    @property
    def registry_path(self) -> Path:
        return self.data_path / "dataset_registry.json"

    @property
    def index_dir(self) -> str:
        return str(self.index_path)

    @property
    def registry_file(self) -> str:
        return str(self.registry_path)

    @property
    def uploads_dir(self) -> str:
        return str(self.upload_path)

    @property
    def corpus_dir(self) -> str:
        return str(self.corpus_path)

    @property
    def eval_path(self) -> Path:
        return self.data_path / "eval"



settings = Settings()
