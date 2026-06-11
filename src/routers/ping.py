import logging

from fastapi import APIRouter, Request

from src.config import get_settings
from src.schemas.api.health import HealthResponse, ServiceStatus
from src.services.ollama.client import OllamaClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/ping")
def ping():
    """
    Simplest possible check — is the API process alive?
    Returns immediately, no DB or external calls.
    Use this for load balancer health checks.
    """
    return {"ping": "pong"}


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """
    Deep health check — verifies every downstream service.
    Like Spring Actuator /actuator/health with show-details=always.
    """
    settings = get_settings()
    services = {}

    # ── Check PostgreSQL ──────────────────────────────────────────
    try:
        database = request.app.state.database
        with database.get_session() as session:
            from sqlalchemy import text

            session.execute(text("SELECT 1"))
        services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services["database"] = ServiceStatus(status="unhealthy", message=str(e))

    # ── Check Ollama ──────────────────────────────────────────────
    try:
        ollama = OllamaClient(host=settings.ollama_host)
        if ollama.is_healthy():
            models = ollama.get_available_models()
            services["ollama"] = ServiceStatus(
                status="healthy", message=f"Available models: {', '.join(models) or 'none pulled yet'}"
            )
        else:
            services["ollama"] = ServiceStatus(status="unhealthy", message="Ollama server not reachable")
    except Exception as e:
        services["ollama"] = ServiceStatus(status="unhealthy", message=str(e))

    # Overall status — degraded if any service is unhealthy
    overall = "ok" if all(s.status == "healthy" for s in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
