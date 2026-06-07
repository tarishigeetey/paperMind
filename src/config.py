from typing import List, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # ignore unknown .env vars
        frozen=True,  # immutable after creation — thread safe
        env_nested_delimiter="__",  # AIRFLOW__CORE__EXECUTOR maps to nested config
    )


class Settings(DefaultSettings):
    """Application settings."""

    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "rag-api"

    # PostgreSQL
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # OpenSearch
    opensearch_host: str = "http://localhost:9200"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_models: Union[str, List[str]] = Field(default=["llama3.2:1b"])
    ollama_default_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    @field_validator("ollama_models", mode="before")
    @classmethod
    def parse_ollama_models(cls, v):
        """Parse comma-separated string into list — env vars come in as strings."""
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v


def get_settings() -> Settings:
    """Get application settings. Called wherever settings are needed."""
    return Settings()
