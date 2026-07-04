# src/services/pgvector/client.py
#
# Episode 9.2 — pgvector replaces OpenSearch
#
# Java analogy: swapping Hibernate Search / Elasticsearch
# for native PostgreSQL full-text + pgvector.
# Same DAO interface, different backend.
#
# BUG FIX (ep9.2): `:embedding::vector` caused psycopg2 SyntaxError
# because `::` conflicts with SQLAlchemy's `:param` binding syntax.
# Fix: use `CAST(:embedding AS vector)` — standard SQL, no ambiguity.

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


class PgVectorClient:
    """
    PostgreSQL + pgvector vector store client.
    Drop-in replacement for OpenSearchClient — identical public interface.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.table_name = "paper_chunks"
        self.engine: Optional[Engine] = None

        db_url = settings.postgres_database_url
        self._db_url = (
            db_url
            if db_url.startswith("postgresql+psycopg2://")
            else db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        )
        logger.info(f"PgVectorClient initialized (table: {self.table_name})")

    def _get_engine(self) -> Engine:
        if not self.engine:
            self.engine = create_engine(
                self._db_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self.engine

    def health_check(self) -> bool:
        try:
            with self._get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
                result = conn.execute(text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"))
                if result.scalar() == 0:
                    logger.warning("pgvector extension not installed")
                return True
        except Exception as e:
            logger.error(f"PgVector health check failed: {e}")
            return False

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """
        Create paper_chunks table, indexes, and tsvector trigger.
        Java analogy: Flyway migration — idempotent, safe to re-run.
        """
        try:
            with self._get_engine().connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                conn.commit()

                if force:
                    conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}"))
                    conn.commit()

                conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id              BIGSERIAL PRIMARY KEY,
                        chunk_id        TEXT UNIQUE,
                        arxiv_id        TEXT NOT NULL,
                        title           TEXT,
                        chunk_text      TEXT NOT NULL,
                        chunk_index     INTEGER DEFAULT 0,
                        categories      TEXT[],
                        published_date  TEXT,
                        authors         TEXT,
                        abstract        TEXT,
                        pdf_url         TEXT,
                        embedding       vector({EMBEDDING_DIM}),
                        tsv             TSVECTOR,
                        created_at      TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                )
                conn.commit()

                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
                """)
                )
                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tsv
                    ON {self.table_name} USING GIN (tsv)
                """)
                )
                conn.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_arxiv_id
                    ON {self.table_name} (arxiv_id)
                """)
                )
                conn.commit()

                conn.execute(
                    text(f"""
                    CREATE OR REPLACE FUNCTION update_{self.table_name}_tsv()
                    RETURNS trigger AS $$
                    BEGIN
                        NEW.tsv := to_tsvector('english',
                            COALESCE(NEW.title, '') || ' ' ||
                            COALESCE(NEW.abstract, '') || ' ' ||
                            COALESCE(NEW.chunk_text, '')
                        );
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                )
                conn.execute(
                    text(f"""
                    DROP TRIGGER IF EXISTS trig_update_{self.table_name}_tsv
                    ON {self.table_name}
                """)
                )
                conn.execute(
                    text(f"""
                    CREATE TRIGGER trig_update_{self.table_name}_tsv
                    BEFORE INSERT OR UPDATE ON {self.table_name}
                    FOR EACH ROW EXECUTE FUNCTION update_{self.table_name}_tsv()
                """)
                )
                conn.commit()
                logger.info(f"pgvector table and indexes ready: {self.table_name}")

            return {"hybrid_index": True, "rrf_pipeline": True}

        except Exception as e:
            logger.error(f"Error setting up pgvector table: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        try:
            with self._get_engine().connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar()
                return {"index_name": self.table_name, "exists": True, "document_count": count}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"index_name": self.table_name, "exists": False, "document_count": 0}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        try:
            if query_embedding and use_hybrid:
                return self._search_hybrid(query, query_embedding, size, categories, min_score)
            elif query_embedding:
                return self._search_vector(query_embedding, size, categories)
            else:
                return self._search_fulltext(query, size, from_, categories, latest)
        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_fulltext(self, query, size, from_, categories, latest) -> Dict[str, Any]:
        try:
            with self._get_engine().connect() as conn:
                params: Dict[str, Any] = {"query": query, "size": size, "offset": from_}
                cat_filter = ""
                if categories:
                    cat_filter = "AND categories && :categories"
                    params["categories"] = categories

                order = "published_date DESC" if latest else "rank DESC"

                rows = conn.execute(
                    text(f"""
                    SELECT
                        chunk_id, arxiv_id, title, chunk_text, chunk_index,
                        categories, published_date, authors, abstract, pdf_url,
                        ts_rank(tsv, plainto_tsquery('english', :query)) AS rank
                    FROM {self.table_name}
                    WHERE tsv @@ plainto_tsquery('english', :query)
                    {cat_filter}
                    ORDER BY {order}
                    LIMIT :size OFFSET :offset
                """),
                    params,
                ).fetchall()

                hits = []
                for row in rows:
                    hits.append(
                        {
                            "chunk_id": row.chunk_id,
                            "arxiv_id": row.arxiv_id,
                            "title": row.title,
                            "chunk_text": row.chunk_text,
                            "chunk_index": row.chunk_index,
                            "categories": row.categories or [],
                            "published_date": row.published_date,
                            "authors": row.authors or "",
                            "abstract": row.abstract,
                            "pdf_url": row.pdf_url,
                            "score": float(row.rank) if row.rank else 0.0,
                        }
                    )
                logger.info(f"Full-text search '{query[:50]}' → {len(hits)} results")
                return {"total": len(hits), "hits": hits}
        except Exception as e:
            logger.error(f"Full-text search error: {e}")
            return {"total": 0, "hits": []}

    def _search_vector(self, query_embedding, size, categories) -> Dict[str, Any]:
        """
        Pure vector search using pgvector cosine similarity.
        FIX: CAST(:embedding AS vector) instead of :embedding::vector
        """
        try:
            with self._get_engine().connect() as conn:
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                params: Dict[str, Any] = {"embedding": embedding_str, "size": size}
                cat_filter = ""
                if categories:
                    cat_filter = "AND categories && :categories"
                    params["categories"] = categories

                rows = conn.execute(
                    text(f"""
                    SELECT
                        chunk_id, arxiv_id, title, chunk_text, chunk_index,
                        categories, published_date, authors, abstract, pdf_url,
                        1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM {self.table_name}
                    WHERE embedding IS NOT NULL
                    {cat_filter}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :size
                """),
                    params,
                ).fetchall()

                hits = [self._row_to_hit(row) for row in rows]
                logger.info(f"Vector search → {len(hits)} results")
                return {"total": len(hits), "hits": hits}
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def _search_hybrid(self, query, query_embedding, size, categories, min_score) -> Dict[str, Any]:
        """
        Hybrid search: full-text + vector with RRF scoring.
        FIX: CAST(:embedding AS vector) instead of :embedding::vector
        Java analogy: two ranked subqueries merged with RRF scoring formula.
        """
        try:
            with self._get_engine().connect() as conn:
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                params: Dict[str, Any] = {
                    "query": query,
                    "embedding": embedding_str,
                    "size": size,
                    "min_score": min_score,
                }
                cat_filter = ""
                if categories:
                    cat_filter = "AND categories && :categories"
                    params["categories"] = categories

                rows = conn.execute(
                    text(f"""
                    WITH
                    fts_ranked AS (
                        SELECT
                            chunk_id,
                            ROW_NUMBER() OVER (
                                ORDER BY ts_rank(tsv, plainto_tsquery('english', :query)) DESC
                            ) AS fts_rank
                        FROM {self.table_name}
                        WHERE tsv @@ plainto_tsquery('english', :query)
                        {cat_filter}
                        LIMIT {size * 5}
                    ),
                    vec_ranked AS (
                        SELECT
                            chunk_id,
                            ROW_NUMBER() OVER (
                                ORDER BY embedding <=> CAST(:embedding AS vector)
                            ) AS vec_rank
                        FROM {self.table_name}
                        WHERE embedding IS NOT NULL
                        {cat_filter}
                        LIMIT {size * 5}
                    ),
                    rrf_scores AS (
                        SELECT
                            COALESCE(f.chunk_id, v.chunk_id) AS chunk_id,
                            COALESCE(1.0/(60 + f.fts_rank), 0) +
                            COALESCE(1.0/(60 + v.vec_rank), 0) AS rrf_score
                        FROM fts_ranked f
                        FULL OUTER JOIN vec_ranked v USING (chunk_id)
                    )
                    SELECT
                        c.chunk_id, c.arxiv_id, c.title, c.chunk_text, c.chunk_index,
                        c.categories, c.published_date, c.authors, c.abstract, c.pdf_url,
                        r.rrf_score AS score
                    FROM rrf_scores r
                    JOIN {self.table_name} c ON c.chunk_id = r.chunk_id
                    WHERE r.rrf_score >= :min_score
                    ORDER BY r.rrf_score DESC
                    LIMIT :size
                """),
                    params,
                ).fetchall()

                hits = [self._row_to_hit(row) for row in rows]
                logger.info(f"Hybrid RRF search '{query[:50]}' → {len(hits)} results")
                return {"total": len(hits), "hits": hits}
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return {"total": 0, "hits": []}

    def _row_to_hit(self, row) -> Dict[str, Any]:
        return {
            "chunk_id": row.chunk_id,
            "arxiv_id": row.arxiv_id,
            "title": row.title,
            "chunk_text": row.chunk_text,
            "chunk_index": row.chunk_index,
            "categories": row.categories or [],
            "published_date": row.published_date,
            "authors": row.authors or "",
            "abstract": row.abstract,
            "pdf_url": row.pdf_url,
            "score": float(row.score) if row.score else 0.0,
        }

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Index a single chunk with its embedding.
        FIX: CAST(:embedding AS vector) instead of :embedding::vector
        """
        try:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            chunk_id = chunk_data.get("chunk_id") or f"{chunk_data.get('arxiv_id')}_{chunk_data.get('chunk_index', 0)}"

            with self._get_engine().connect() as conn:
                conn.execute(
                    text(f"""
                    INSERT INTO {self.table_name}
                        (chunk_id, arxiv_id, title, chunk_text, chunk_index,
                         categories, published_date, authors, abstract, pdf_url, embedding)
                    VALUES
                        (:chunk_id, :arxiv_id, :title, :chunk_text, :chunk_index,
                         :categories, :published_date, :authors, :abstract, :pdf_url,
                         CAST(:embedding AS vector))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chunk_text     = EXCLUDED.chunk_text,
                        embedding      = EXCLUDED.embedding,
                        title          = EXCLUDED.title,
                        categories     = EXCLUDED.categories,
                        published_date = EXCLUDED.published_date
                """),
                    {
                        "chunk_id": chunk_id,
                        "arxiv_id": chunk_data.get("arxiv_id"),
                        "title": chunk_data.get("title"),
                        "chunk_text": chunk_data.get("chunk_text", ""),
                        "chunk_index": chunk_data.get("chunk_index", 0),
                        "categories": chunk_data.get("categories"),
                        "published_date": chunk_data.get("published_date"),
                        "authors": chunk_data.get("authors"),
                        "abstract": chunk_data.get("abstract"),
                        "pdf_url": chunk_data.get("pdf_url"),
                        "embedding": embedding_str,
                    },
                )
                conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk index multiple chunks.
        FIX: CAST(:embedding AS vector) instead of :embedding::vector
        Java analogy: JDBC executeBatch
        """
        success = 0
        failed = 0

        try:
            with self._get_engine().connect() as conn:
                for chunk in chunks:
                    try:
                        chunk_data = chunk["chunk_data"]
                        embedding = chunk["embedding"]
                        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                        chunk_id = (
                            chunk_data.get("chunk_id") or f"{chunk_data.get('arxiv_id')}_{chunk_data.get('chunk_index', 0)}"
                        )

                        conn.execute(
                            text(f"""
                            INSERT INTO {self.table_name}
                                (chunk_id, arxiv_id, title, chunk_text, chunk_index,
                                 categories, published_date, authors, abstract, pdf_url, embedding)
                            VALUES
                                (:chunk_id, :arxiv_id, :title, :chunk_text, :chunk_index,
                                 :categories, :published_date, :authors, :abstract, :pdf_url,
                                 CAST(:embedding AS vector))
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                chunk_text     = EXCLUDED.chunk_text,
                                embedding      = EXCLUDED.embedding,
                                title          = EXCLUDED.title,
                                categories     = EXCLUDED.categories,
                                published_date = EXCLUDED.published_date
                        """),
                            {
                                "chunk_id": chunk_id,
                                "arxiv_id": chunk_data.get("arxiv_id"),
                                "title": chunk_data.get("title"),
                                "chunk_text": chunk_data.get("chunk_text", ""),
                                "chunk_index": chunk_data.get("chunk_index", 0),
                                "categories": chunk_data.get("categories"),
                                "published_date": chunk_data.get("published_date"),
                                "authors": chunk_data.get("authors"),
                                "abstract": chunk_data.get("abstract"),
                                "pdf_url": chunk_data.get("pdf_url"),
                                "embedding": embedding_str,
                            },
                        )
                        success += 1
                    except Exception as e:
                        logger.error(f"Error indexing chunk: {e}")
                        failed += 1

                conn.commit()

            logger.info(f"Bulk indexed {success} chunks, {failed} failed")
            return {"success": success, "failed": failed}

        except Exception as e:
            logger.error(f"Bulk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        try:
            with self._get_engine().connect() as conn:
                result = conn.execute(text(f"DELETE FROM {self.table_name} WHERE arxiv_id = :arxiv_id"), {"arxiv_id": arxiv_id})
                conn.commit()
                logger.info(f"Deleted {result.rowcount} chunks for {arxiv_id}")
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        try:
            with self._get_engine().connect() as conn:
                rows = conn.execute(
                    text(f"""
                    SELECT chunk_id, arxiv_id, title, chunk_text, chunk_index,
                           categories, published_date, authors, abstract, pdf_url
                    FROM {self.table_name}
                    WHERE arxiv_id = :arxiv_id
                    ORDER BY chunk_index ASC
                """),
                    {"arxiv_id": arxiv_id},
                ).fetchall()
                return [self._row_to_hit(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
