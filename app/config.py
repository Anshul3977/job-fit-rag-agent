"""Runtime configuration. Environment variables override .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_database_url(url: str) -> str:
    """Use the psycopg v3 driver when a bare Postgres URL is provided."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "job-fit-rag-agent"
    version: str = "1.0.0"
    env: str = "dev"
    port: int = 8741
    log_level: str = "INFO"

    database_url: str = "sqlite:///./data/jobfit.db"
    vector_backend: str = "auto"
    chroma_dir: str = "./data/chroma"

    embedding_backend: str = "fastembed"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = "./data/models"

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = 25.0
    llm_provider: str = "auto"

    max_upload_bytes: int = 5_000_000
    max_job_chars: int = 20_000
    rate_limit_fit_per_minute: int = 10
    cors_origins: str = "http://localhost:5174,http://127.0.0.1:5174"
    debug_routes: bool = False
    frontend_dist: str = "frontend/dist"

    @property
    def sqlalchemy_url(self) -> str:
        return normalize_database_url(self.database_url)

    @property
    def resolved_vector_backend(self) -> str:
        if self.vector_backend != "auto":
            return self.vector_backend
        if self.sqlalchemy_url.startswith("postgresql"):
            return "pgvector"
        return "chroma"

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.groq_api_key.strip():
            return "groq"
        return "deterministic"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def data_dir(self) -> Path:
        if self.sqlalchemy_url.startswith("sqlite:///"):
            raw = self.sqlalchemy_url[len("sqlite:///") :]
            if raw and raw != ":memory:":
                return Path(raw).resolve().parent
        return Path("data").resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
