from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status of a single downstream service."""

    status: str = Field(..., description="healthy or unhealthy")
    message: Optional[str] = Field(None, description="Details")


class HealthResponse(BaseModel):
    """
    Full health check response.
    Like a DTO — controls exactly what the API exposes.
    """

    status: str = Field(..., description="ok or degraded")
    version: str
    environment: str
    service_name: str
    services: Optional[Dict[str, ServiceStatus]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "rag-api",
                "services": {"database": {"status": "healthy", "message": "Connected successfully"}},
            }
        }
