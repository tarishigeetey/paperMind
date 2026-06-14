from functools import lru_cache

from src.config import get_settings
from src.services.opensearch.client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client() -> OpenSearchClient:
    """
    Factory — creates a cached OpenSearchClient.

    @lru_cache because the OpenSearch connection is expensive to create.
    One instance shared across the entire application.
    Like a Spring @Bean singleton.
    """
    settings = get_settings()
    return OpenSearchClient(
        host=settings.opensearch.host,
        settings=settings,
    )
