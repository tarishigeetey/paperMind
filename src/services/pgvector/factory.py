# src/services/pgvector/factory.py
#
# Java analogy: @Bean singleton factory — same pattern as opensearch/factory.py
# The app calls make_vector_store_client() — doesn't know if it's OpenSearch or pgvector.

from functools import lru_cache

from src.config import get_settings
from src.services.pgvector.client import PgVectorClient


@lru_cache(maxsize=1)
def make_pgvector_client() -> PgVectorClient:
    """
    Create and return a singleton PgVectorClient instance.
    Java analogy: @Bean with @Scope("singleton").
    """
    settings = get_settings()
    return PgVectorClient(settings=settings)
