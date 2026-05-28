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
    llm_analysis_sample_rows: int = 5000
    llm_analysis_preview_rows: int = 5
    llm_analysis_max_columns: int = 80
    llm_rag_max_chunks: int = 6
    llm_rag_max_chars_per_chunk: int = 1200

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



settings = Settings()
