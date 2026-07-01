from typing import Optional

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient


def make_embeddings_service(
    settings: Optional[Settings] = None,
) -> JinaEmbeddingsClient:
    """
    Factory — creates embeddings service.
    Creates new instance each time — avoids closed client issues.
    """
    if settings is None:
        settings = get_settings()
    return JinaEmbeddingsClient(api_key=settings.jina_api_key)


def make_embeddings_client(
    settings: Optional[Settings] = None,
) -> JinaEmbeddingsClient:
    """
    Alias for make_embeddings_service.
    Called by HybridIndexingService factory.
    """
    if settings is None:
        settings = get_settings()
    return JinaEmbeddingsClient(api_key=settings.jina_api_key)
