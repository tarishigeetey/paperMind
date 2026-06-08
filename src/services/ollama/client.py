import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    HTTP client for the Ollama local LLM server.
    Week 1: health check only.
    Week 5: full chat completions.

    Like a Spring @Service wrapping a RestTemplate.
    """

    def __init__(self, host: str, timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        """
        Check if Ollama server is reachable.
        GET /api/tags returns list of downloaded models.
        """
        try:
            response = httpx.get(
                f"{self.host}/api/tags",
                timeout=5.0,  # short timeout for health check
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Return list of models downloaded in Ollama."""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Download a model if not already present.
        Called at startup to ensure the default model exists.
        """
        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            response = httpx.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                timeout=self.timeout,  # pulling can take minutes
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
