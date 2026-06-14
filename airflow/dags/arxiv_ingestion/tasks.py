import asyncio
import logging
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Optional, Tuple

sys.path.insert(0, "/opt/airflow")

from sqlalchemy import text
from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any, Any]:
    """
    Initialise all services once — cached for the process lifetime.
    Week 3 adds opensearch_client to the tuple.
    """
    logger.info("Initialising services")
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    opensearch_client = make_opensearch_client()
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)
    logger.info("All services ready")
    return arxiv_client, pdf_parser, database, metadata_fetcher, opensearch_client


async def run_paper_ingestion_pipeline(
    target_date: str,
    max_results: Optional[int] = None,
    process_pdfs: bool = True,
    index_to_opensearch: bool = False,
) -> dict:
    """Async pipeline wrapper."""
    arxiv_client, _pdf_parser, database, metadata_fetcher, _opensearch_client = get_cached_services()

    if max_results is None:
        max_results = arxiv_client.max_results

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def setup_environment(**context):
    """Task 1 — verify all services before pipeline starts."""
    logger.info("Setting up environment")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("✅ Database connection verified")

        if opensearch_client.health_check():
            logger.info("✅ OpenSearch connection verified")
            if opensearch_client.create_index(force=False):
                logger.info("OpenSearch index created")
            else:
                logger.info("OpenSearch index already exists")
        else:
            logger.warning("⚠️ OpenSearch not healthy — indexing will be skipped")

        logger.info(f"✅ arXiv client ready: {arxiv_client.base_url}")
        return {"status": "success"}

    except Exception as e:
        raise Exception(f"Environment setup failed: {e}")


def fetch_daily_papers(**context):
    """
    Task 2 — fetch papers from arXiv and store to PostgreSQL.
    Week 3 note: index_to_opensearch=False here.
    Indexing is a separate dedicated task — cleaner separation.
    """
    logger.info("Starting daily paper fetch")

    try:
        execution_date = context["ds"]
        target_dt = datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")
        logger.info(f"Fetching papers for: {target_date}")

        results = asyncio.run(
            run_paper_ingestion_pipeline(
                target_date=target_date,
                process_pdfs=True,
                index_to_opensearch=False,  # dedicated task handles this
            )
        )

        context["task_instance"].xcom_push(key="fetch_results", value=results)
        return results

    except Exception as e:
        raise Exception(f"Daily fetch failed: {e}")


def index_papers_to_opensearch(**context):
    """
    Task 3 — index today's papers from PostgreSQL into OpenSearch.

    Week 3 replaces create_opensearch_placeholders() with this real task.

    Why separate from fetch_daily_papers?
    Single Responsibility — one task fetches, one task indexes.
    If indexing fails, you can rerun just this task without re-fetching.
    If OpenSearch is down, fetching still succeeds and papers are in PostgreSQL.
    """
    logger.info("Starting OpenSearch indexing")

    try:
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        if not fetch_results:
            logger.warning("No fetch results — skipping indexing")
            return {"status": "skipped", "papers_indexed": 0}

        papers_stored = fetch_results.get("papers_stored", 0)

        if papers_stored == 0:
            logger.info("No papers stored — skipping indexing")
            return {"status": "skipped", "papers_indexed": 0}

        _arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        if not opensearch_client.health_check():
            logger.error("OpenSearch not healthy — skipping indexing")
            return {"status": "failed", "papers_indexed": 0}

        indexed_count = 0
        failed_count = 0

        with database.get_session() as session:
            paper_repo = PaperRepository(session)

            # Fetch today's papers from PostgreSQL
            result = session.execute(
                text("""
                    SELECT id FROM papers
                    WHERE DATE(created_at) = CURRENT_DATE
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"limit": papers_stored},
            )
            paper_ids = [row.id for row in result.fetchall()]
            logger.info(f"Found {len(paper_ids)} papers to index")

            for paper_id in paper_ids:
                try:
                    paper = paper_repo.get_by_id(paper_id)
                    if not paper:
                        continue

                    # Build the OpenSearch document
                    paper_doc = {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": (", ".join(paper.authors) if isinstance(paper.authors, list) else str(paper.authors)),
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url,
                        "published_date": (
                            paper.published_date.isoformat()
                            if hasattr(paper.published_date, "isoformat")
                            else str(paper.published_date)
                        ),
                        "raw_text": paper.raw_text or "",
                        "created_at": (
                            paper.created_at.isoformat() if hasattr(paper.created_at, "isoformat") else str(paper.created_at)
                        ),
                        "updated_at": (
                            paper.updated_at.isoformat() if hasattr(paper.updated_at, "isoformat") else str(paper.updated_at)
                        ),
                    }

                    if opensearch_client.index_paper(paper_doc):
                        indexed_count += 1
                        logger.debug(f"Indexed: {paper.arxiv_id}")
                    else:
                        failed_count += 1

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error indexing {paper_id}: {e}")

        # Final stats
        try:
            final_stats = opensearch_client.get_index_stats()
            total_docs = final_stats.get("document_count", 0)
        except Exception:
            total_docs = "unknown"

        result = {
            "status": "completed",
            "papers_indexed": indexed_count,
            "indexing_failures": failed_count,
            "total_documents_in_index": total_docs,
        }

        logger.info("=" * 50)
        logger.info("OpenSearch Indexing Summary")
        logger.info(f"  Indexed:  {indexed_count}")
        logger.info(f"  Failed:   {failed_count}")
        logger.info(f"  Total in index: {total_docs}")
        logger.info("=" * 50)

        return result

    except Exception as e:
        logger.error(f"OpenSearch indexing failed: {e}")
        return {"status": "error", "papers_indexed": 0}


def generate_daily_report(**context):
    """Task 4 — collect results from all tasks and log summary."""
    logger.info("Generating daily report")

    try:
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")
        opensearch_results = context["task_instance"].xcom_pull(task_ids="index_papers_to_opensearch")

        report = {
            "date": context["ds"],
            "papers": {
                "fetched": fetch_results.get("papers_fetched", 0) if fetch_results else 0,
                "pdfs_downloaded": fetch_results.get("pdfs_downloaded", 0) if fetch_results else 0,
                "pdfs_parsed": fetch_results.get("pdfs_parsed", 0) if fetch_results else 0,
                "stored": fetch_results.get("papers_stored", 0) if fetch_results else 0,
            },
            "processing": {
                "time_seconds": fetch_results.get("processing_time", 0) if fetch_results else 0,
                "errors": len(fetch_results.get("errors", [])) if fetch_results else 0,
            },
            "opensearch": {
                "indexed": opensearch_results.get("papers_indexed", 0) if opensearch_results else 0,
                "failures": opensearch_results.get("indexing_failures", 0) if opensearch_results else 0,
                "total_in_index": opensearch_results.get("total_documents_in_index", 0) if opensearch_results else 0,
                "status": opensearch_results.get("status", "unknown") if opensearch_results else "unknown",
            },
        }

        logger.info("=" * 50)
        logger.info("DAILY REPORT")
        logger.info("=" * 50)
        logger.info(f"Date:              {report['date']}")
        logger.info(f"Papers fetched:    {report['papers']['fetched']}")
        logger.info(f"PDFs downloaded:   {report['papers']['pdfs_downloaded']}")
        logger.info(f"PDFs parsed:       {report['papers']['pdfs_parsed']}")
        logger.info(f"Papers stored:     {report['papers']['stored']}")
        logger.info(f"Processing time:   {report['processing']['time_seconds']:.1f}s")
        logger.info(f"Errors:            {report['processing']['errors']}")
        logger.info(f"OpenSearch indexed:{report['opensearch']['indexed']}")
        logger.info(f"OpenSearch total:  {report['opensearch']['total_in_index']}")
        logger.info("=" * 50)

        return report

    except Exception as e:
        raise Exception(f"Report generation failed: {e}")
