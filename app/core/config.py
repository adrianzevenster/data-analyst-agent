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
