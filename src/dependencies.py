from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.ollama.client import OllamaClient


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """
    Pull OpenSearch client from app.state.
    Stored there at startup in lifespan().
    Same pattern as get_database().
    """
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    return request.app.state.pdf_parser

def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service

def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client

# ── Type aliases ──────────────────────────────────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]