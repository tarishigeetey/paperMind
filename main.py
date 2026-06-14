import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.factory import make_database
from src.routers import papers, ping, search
from src.services.arxiv.factory import make_arxiv_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App startup and shutdown.
    Week 3 adds full OpenSearch initialisation.
    """
    logger.info("Starting PaperMind API...")
    settings = get_settings()
    app.state.settings = settings

    # ── PostgreSQL ─────────────────────────────────────────────────
    database = make_database()
    app.state.database = database
    logger.info("PostgreSQL connected")

    # ── OpenSearch ─────────────────────────────────────────────────
    # Week 3 — no longer a placeholder, real client
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    if opensearch_client.health_check():
        logger.info("OpenSearch connected")

        # Create index if it doesn't exist
        # force=False means skip if already exists — idempotent
        if opensearch_client.create_index(force=False):
            logger.info("OpenSearch index created")
        else:
            logger.info("OpenSearch index already exists")

        # Log how many papers are already indexed
        stats = opensearch_client.get_index_stats()
        doc_count = stats.get("document_count", 0)
        logger.info(f"OpenSearch ready: {doc_count} documents indexed")
    else:
        logger.warning("OpenSearch not reachable — search features degraded")

    # ── Other services ─────────────────────────────────────────────
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    logger.info("arXiv client and PDF parser ready")

    logger.info("PaperMind API ready ✅")
    yield

    # ── Shutdown ───────────────────────────────────────────────────
    database.teardown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperMind API",
        description="Production RAG system for academic research papers",
        version=os.getenv("APP_VERSION", "0.1.0"),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Week 3 adds search router
    app.include_router(ping.router, prefix="/api/v1")
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
