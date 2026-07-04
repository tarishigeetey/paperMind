from typing import Optional

from src.config import Settings, get_settings
from src.services.embeddings.factory import make_embeddings_client

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker


def make_hybrid_indexing_service(
    settings: Optional[Settings] = None,
    opensearch_host: Optional[str] = None,
) -> HybridIndexingService:
    """
    Factory — creates HybridIndexingService with all dependencies.

    Episode 9.2: VECTOR_STORE env var switches between pgvector and OpenSearch.
    Java analogy: @ConditionalOnProperty — same bean, different implementation
    injected based on configuration.

    VECTOR_STORE=pgvector  → PgVectorClient (default, prod)
    VECTOR_STORE=opensearch → OpenSearchClient (legacy, local dev)
    """
    if settings is None:
        settings = get_settings()

    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,
    )

    embeddings_client = make_embeddings_client(settings)

    # Dynamic vector store selection
    vector_store = getattr(settings, "vector_store", "pgvector").lower()

    if vector_store == "opensearch":
        from src.services.opensearch.factory import make_opensearch_client_fresh

        vector_client = make_opensearch_client_fresh(settings, host=opensearch_host)
    else:
        # pgvector is default
        from src.services.pgvector.client import PgVectorClient

        vector_client = PgVectorClient(settings=settings)

    return HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=vector_client,  # same param name — no change in HybridIndexingService
    )
