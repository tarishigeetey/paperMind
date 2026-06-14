import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root from this file's location
# src/config.py → src/ → project root
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    """
    Base settings class all other settings inherit from.
    Loads from both local .env and the resolved project root .env.
    frozen=True makes all settings immutable after creation.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    """arXiv API client settings — reads ARXIV__* env vars."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1
    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Create cache directory if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    """PDF parser settings — reads PDF_PARSER__* env vars."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class OpenSearchSettings(BaseConfigSettings):
    """
    OpenSearch settings — reads OPENSEARCH__* env vars.
    Week 3 addition.
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    max_text_size: int = 1_000_000  # 1MB max text per document


class Settings(BaseConfigSettings):
    """Root application settings."""

    app_version: str = "0.1.0"
    debug: bool = True

    # Literal type — only these three values accepted
    # Pydantic validates at startup — typo in .env crashes immediately
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    # PostgreSQL
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    # Ollama — simplified in Week 3 (list → single model)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"  # single model, not a list
    ollama_timeout: int = 300

    # Nested settings — each reads their own env prefix
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Fail fast if database URL format is wrong."""
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


def get_settings() -> Settings:
    return Settings()
