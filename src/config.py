from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
    )


class ArxivSettings(DefaultSettings):
    """
    arXiv API client settings.
    Like a @ConfigurationProperties(prefix="arxiv") in Spring.
    """

    base_url: str = "https://export.arxiv.org/api/query"
    namespaces: dict = Field(
        default={
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
    )
    pdf_cache_dir: str = "./data/arxiv_pdfs"  # local cache for downloaded PDFs
    rate_limit_delay: float = 3.0  # arXiv asks for 3s between requests
    timeout_seconds: int = 30
    max_results: int = 100
    search_category: str = "cs.AI"  # which arXiv category to fetch


class PDFParserSettings(DefaultSettings):
    """
    Docling PDF parser settings.
    Like a @ConfigurationProperties(prefix="pdf-parser") in Spring.
    """

    max_pages: int = 30  # skip PDFs longer than this
    max_file_size_mb: int = 20  # skip PDFs larger than this
    do_ocr: bool = False  # OCR is very slow — off by default
    do_table_structure: bool = True


class Settings(DefaultSettings):
    """Application settings — root config object."""

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
    ollama_models: List[str] = Field(default=["llama3.2:1b"])
    ollama_default_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    # Week 2 — nested settings objects
    # Field(default_factory=...) creates a fresh instance each time
    # Like @Autowired ArxivConfig arxivConfig in Spring
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)

    @field_validator("ollama_models", mode="before")
    @classmethod
    def parse_ollama_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(",") if model.strip()]
        return v


def get_settings() -> Settings:
    return Settings()
