from functools import lru_cache

from src.config import get_settings
from src.services.embeddings.jina_client import JinaEmbeddingsClient

@lru_cache(maxsize=1)
def make_embeddings_client() -> JinaEmbeddingsClient:
    """
    Factory — creates a cached JinaEmbeddingsClient.

    @lru_cache = singleton.
    The HTTP client inside JinaEmbeddingsClient is reused
    across all requests — no connection overhead per call.

    Like a @Bean singleton in Spring.
    """
    settings = get_settings()

    if not settings.jina_api_key:
        raise ValueError(
            "JINA_API_KEY not set in .env. "
            "Get a free key at jina.ai"
        )

    return JinaEmbeddingsClient(api_key=settings.jina_api_key)