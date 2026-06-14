import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from src.config import Settings, get_settings
from src.services.opensearch.index_config import (
    ARXIV_PAPERS_INDEX,
    ARXIV_PAPERS_MAPPING,
)
from src.services.opensearch.query_builder import PaperQueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """
    Service for all OpenSearch operations.
    Like a Spring @Service wrapping ElasticsearchOperations.

    Responsibilities:
    - Index lifecycle (create, check exists)
    - Document indexing (single + bulk)
    - BM25 search with query builder
    - Health and stats monitoring
    """

    def __init__(
        self,
        host: str = "http://localhost:9200",
        settings: Optional[Settings] = None,
    ):
        self.host = host
        self.settings = settings or get_settings()

        # Low-level OpenSearch Python client
        # http_compress=True → gzip requests/responses (faster for large docs)
        # use_ssl=False → dev only, production would use TLS
        self.client = OpenSearch(
            hosts=[host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

        # Use index name from settings, fall back to constant
        self.index_name = self.settings.opensearch.index_name or ARXIV_PAPERS_INDEX
        logger.info(f"OpenSearch client initialised: {host}")

    # ── Index management ───────────────────────────────────────────

    def create_index(self, force: bool = False) -> bool:
        """
        Create the arxiv-papers index with mappings.

        force=True deletes and recreates if index exists.
        force=False skips creation if index already exists.

        Called at app startup in lifespan() — idempotent, safe to run always.
        Like Flyway/Liquibase migrations — runs every startup, skips if done.
        """
        try:
            if self.client.indices.exists(index=self.index_name):
                if force:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index already exists: {self.index_name}")
                    return False

            response = self.client.indices.create(
                index=self.index_name,
                body=ARXIV_PAPERS_MAPPING,
            )

            if response.get("acknowledged"):
                logger.info(f"Created index: {self.index_name}")
                return True
            else:
                logger.error(f"Index creation not acknowledged: {response}")
                return False

        except RequestError as e:
            logger.error(f"Error creating index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False

    # ── Document indexing ──────────────────────────────────────────

    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        Index a single paper document.

        Uses arxiv_id as the document ID — upsert behaviour.
        If the paper already exists, it's updated. No duplicates.
        refresh=True makes the document immediately searchable.

        Like session.merge() in JPA — insert or update.
        """
        try:
            if "arxiv_id" not in paper_data:
                logger.error("Missing arxiv_id — cannot index paper")
                return False

            # Add timestamps if missing
            now = datetime.now(timezone.utc).isoformat()
            if "created_at" not in paper_data:
                paper_data["created_at"] = now
            if "updated_at" not in paper_data:
                paper_data["updated_at"] = now

            # Convert authors list to string
            # PostgreSQL stores ["Vaswani", "Shazeer"]
            # OpenSearch text field expects "Vaswani, Shazeer"
            if isinstance(paper_data.get("authors"), list):
                paper_data["authors"] = ", ".join(paper_data["authors"])

            response = self.client.index(
                index=self.index_name,
                id=paper_data["arxiv_id"],  # doc ID = arxiv_id
                body=paper_data,
                refresh=True,  # immediately searchable after indexing
            )

            if response.get("result") in ["created", "updated"]:
                logger.debug(f"Indexed paper: {paper_data['arxiv_id']}")
                return True
            else:
                logger.error(f"Unexpected index response: {response}")
                return False

        except Exception as e:
            logger.error(f"Error indexing paper {paper_data.get('arxiv_id', 'unknown')}: {e}")
            return False

    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple papers.
        Returns counts of successes and failures.

        Note: this calls index_paper() per document.
        Week 4 could optimise with OpenSearch bulk API for large batches.
        For 10-100 papers/day, per-document indexing is fine.
        """
        results = {"success": 0, "failed": 0}

        for paper in papers:
            if self.index_paper(paper):
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(f"Bulk indexing: {results['success']} success, {results['failed']} failed")
        return results

    # ── Search ─────────────────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ) -> Dict[str, Any]:
        """
        Search papers using BM25 via PaperQueryBuilder.

        Returns:
        {
            "total": 42,
            "hits": [
                {
                    "arxiv_id": "2401.00001",
                    "title": "...",
                    "score": 8.42,
                    "highlights": {"abstract": ["...matched text..."]}
                }
            ]
        }
        """

        try:
            # Delegate query construction to PaperQueryBuilder
            # OpenSearchClient doesn't know about query syntax
            # PaperQueryBuilder doesn't know about HTTP or indices
            # Clean separation of concerns
            query_builder = PaperQueryBuilder(
                query=query,
                size=size,
                from_=from_,
                fields=fields,
                categories=categories,
                track_total_hits=track_total_hits,
                latest_papers=latest_papers,
            )
            search_body = query_builder.build()

            response = self.client.search(
                index=self.index_name,
                body=search_body,
            )

            # Parse OpenSearch response into clean dict
            results = {
                "total": response["hits"]["total"]["value"],
                "hits": [],
            }

            for hit in response["hits"]["hits"]:
                paper = hit["_source"]
                paper["score"] = hit["_score"]

                # Add highlights if present
                if "highlight" in hit:
                    paper["highlights"] = hit["highlight"]

                results["hits"].append(paper)

            logger.info(f"Search '{query}' → {results['total']} results")
            return results

        except NotFoundError:
            logger.error(f"Index not found: {self.index_name}")
            return {"total": 0, "hits": [], "error": "Index not found"}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "hits": [], "error": str(e)}

    # ── Monitoring ─────────────────────────────────────────────────

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics for monitoring.
        Called at startup and in the health endpoint.
        Like JPA's EntityManager statistics.
        """
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": self.client.cluster.health(index=self.index_name)["status"],
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> bool:
        """
        Check if OpenSearch cluster is healthy.
        green = all shards assigned
        yellow = some replica shards unassigned (fine for single-node dev)
        red = primary shards missing — data loss risk
        """
        try:
            health = self.client.cluster.health()
            is_healthy = health["status"] in ["green", "yellow"]
            if not is_healthy:
                logger.warning(f"OpenSearch cluster status: {health['status']}")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """Get cluster version and node info."""
        try:
            return self.client.info()
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return None

    def get_cluster_health(self) -> Optional[Dict[str, Any]]:
        """Get detailed cluster health."""
        try:
            return self.client.cluster.health()
        except Exception as e:
            logger.error(f"Error getting cluster health: {e}")
            return None

    def get_index_mapping(self) -> Optional[Dict[str, Any]]:
        """
        Get current index mapping.
        Useful for verifying the mapping was applied correctly.
        Like DESCRIBE TABLE in SQL.
        """
        try:
            mappings = self.client.indices.get_mapping(index=self.index_name)
            if mappings and self.index_name in mappings:
                return mappings[self.index_name].get("mappings", {})
            return {}
        except Exception as e:
            logger.error(f"Error getting mapping: {e}")
            return None

    def get_index_settings(self) -> Optional[Dict[str, Any]]:
        """Get current index settings."""
        try:
            settings = self.client.indices.get_settings(index=self.index_name)
            if settings and self.index_name in settings:
                return settings[self.index_name].get("settings", {})
            return {}
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return None
