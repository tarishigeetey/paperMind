import logging
from typing import List

import httpx

from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """
    Client for Jina AI embeddings API.

    Two methods:
    - embed_passages(): for indexing document chunks (task=retrieval.passage)
    - embed_query():    for search queries (task=retrieval.query)

    Like a Spring @Service wrapping an external REST API.
    Uses async httpx — non-blocking, handles concurrent batch requests.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.jina.ai/v1",
    ):
        if not api_key:
            raise ValueError("Jina API key is required. Get a free key at jina.ai and set JINA_API_KEY in .env")

        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Shared async client — reused across all calls
        # Like a shared RestTemplate bean in Spring
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina embeddings client initialised")

    async def embed_passages(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Embed text passages for indexing.
        task=retrieval.passage → optimised to be found by queries.

        Processes in batches because Jina has a per-request limit.
        batch_size=100 is safe — well within Jina's limits.

        Returns: List of 1024-dimensional float vectors, one per input text.
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1} ({len(batch)} passages)")

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3",
                task="retrieval.passage",  # ← for indexing
                dimensions=1024,
                input=batch,
            )

            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=request_data.model_dump(),
                )
                response.raise_for_status()

                result = JinaEmbeddingResponse(**response.json())

                # data is a list of dicts sorted by index
                # extract just the embedding vectors in order
                batch_embeddings = [item["embedding"] for item in result.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded {len(batch)} passages successfully")

            except httpx.HTTPStatusError as e:
                logger.error(f"Jina API HTTP error: {e.response.status_code} — {e.response.text}")
                raise
            except httpx.TimeoutException:
                logger.error("Jina API request timed out")
                raise
            except Exception as e:
                logger.error(f"Unexpected error embedding passages: {e}")
                raise

        logger.info(f"Embedded {len(texts)} passages total")
        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        task=retrieval.query → optimised to find relevant passages.

        Returns: single 1024-dimensional float vector.

        Called once per search request — fast, single text input.
        """
        request_data = JinaEmbeddingRequest(
            model="jina-embeddings-v3",
            task="retrieval.query",  # ← for searching, not indexing
            dimensions=1024,
            input=[query],  # single query wrapped in list
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=request_data.model_dump(),
            )
            response.raise_for_status()

            result = JinaEmbeddingResponse(**response.json())
            embedding = result.data[0]["embedding"]

            logger.debug(f"Embedded query: '{query[:50]}...' → {len(embedding)} dims")
            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"Jina API error embedding query: {e.response.status_code} — {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    async def close(self):
        """Close the shared HTTP client. Called at app shutdown."""
        await self.client.aclose()

    # ── Async context manager support ─────────────────────────────
    # Allows: async with JinaEmbeddingsClient(key) as client:
    # Like try-with-resources in Java

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
