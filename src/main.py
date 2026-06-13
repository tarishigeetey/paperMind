import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.factory import make_database
from src.routers import papers, ping

# Configure logging once at app level
# Like logging.properties in Spring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    Everything before yield runs on startup.
    Everything after yield runs on shutdown.
    Like @PostConstruct + @PreDestroy in Spring.
    """

    # ── STARTUP ────────────────────────────────────────────────────
    settings = get_settings()
    logger.info(f"Starting {settings.service_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database — creates tables if they don't exist
    # Like Spring's DataSource initialization
    logger.info("Initializing database...")
    database = make_database()

    # Store on app.state — accessible everywhere via request.app.state
    # Like Spring's ApplicationContext holding beans
    app.state.database = database
    app.state.settings = settings

    # Placeholders for services added in later weeks
    app.state.opensearch_service = None  # Week 3
    app.state.llm_service = None  # Week 5

    logger.info(f"✅ {settings.service_name} started successfully")

    yield  # ← app is running, serving requests

    # ── SHUTDOWN ───────────────────────────────────────────────────
    logger.info("Shutting down...")
    database.teardown()
    logger.info("Database connections closed")
    logger.info(f"👋 {settings.service_name} stopped")


def create_app() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI app.
    Like a Spring @Configuration class.
    Separate from module level so it's easy to test —
    tests call create_app() with overrides, not import the module.
    """
    settings = get_settings()

    app = FastAPI(
        title="PaperMind API",
        description="Production RAG system for academic research papers",
        version=settings.app_version,
        lifespan=lifespan,
        # Swagger UI at /docs, ReDoc at /redoc
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS middleware ────────────────────────────────────────────
    # Like Spring's @CrossOrigin — allows browser requests from other origins
    # In dev we allow everything — tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────
    # All routes get /api/v1 prefix
    # Like @RequestMapping("/api/v1") on every controller
    app.include_router(ping.router, prefix="/api/v1")
    app.include_router(papers.router, prefix="/api/v1")

    return app


# Create the app instance — uvicorn imports this
app = create_app()
