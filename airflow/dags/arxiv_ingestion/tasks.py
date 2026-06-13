import asyncio
import logging
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Tuple

# Add project src to Python path
# The Airflow container mounts src/ at /opt/airflow/src
sys.path.insert(0, "/opt/airflow")

from sqlalchemy import text
from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any]:
    """
    Initialize all services once and cache them.

    Why @lru_cache here?
    Airflow runs each task in the same process when using LocalExecutor.
    @lru_cache means services (especially Docling with its PyTorch models)
    are initialised ONCE per Airflow worker process, not once per task run.
    Critical for Docling — loading PyTorch models takes 30 seconds.

    Returns: (arxiv_client, pdf_parser, database, metadata_fetcher)
    """
    logger.info("Initialising services (first time only — cached after this)")

    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()  # loads Docling models here
    database = make_database()
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)

    logger.info("All services initialised and cached")
    return arxiv_client, pdf_parser, database, metadata_fetcher


async def run_paper_ingestion_pipeline(
    target_date: str,
    max_results: int = 10,
    process_pdfs: bool = True,
) -> dict:
    """
    Async wrapper for the full pipeline.
    Airflow task functions are sync — this bridges to async MetadataFetcher.
    """
    _arxiv_client, _pdf_parser, database, metadata_fetcher = get_cached_services()

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
    """
    Task 1 — verify all services are reachable before starting.
    Fails the DAG early if anything is down rather than
    wasting time downloading papers we can't store.
    Like a @BeforeClass setup in JUnit.
    """
    logger.info("Setting up environment for arXiv paper ingestion")

    try:
        arxiv_client, _pdf_parser, database, _metadata_fetcher = get_cached_services()

        # Test database connection
        with database.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("✅ Database connection verified")

        logger.info(f"✅ arXiv client ready: {arxiv_client.base_url}")
        logger.info(f"✅ Search category: {arxiv_client.search_category}")
        logger.info("✅ PDF parser service ready")

        return {"status": "success", "message": "Environment setup completed"}

    except Exception as e:
        error_msg = f"Environment setup failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def fetch_daily_papers(**context):
    """
    Task 2 — the main pipeline task.

    Calculates yesterday's date and fetches papers submitted that day.
    Why yesterday? arXiv papers submitted today may not be fully indexed yet.

    context["ds"] is Airflow's execution date in YYYY-MM-DD format.
    We subtract one day to get papers that are definitely available.
    """
    logger.info("Starting daily arXiv paper fetch")

    try:
        # Airflow provides execution_date via context["ds"]
        # Format: "2024-01-15" (YYYY-MM-DD)
        execution_date = context["ds"]
        execution_dt = datetime.strptime(execution_date, "%Y-%m-%d")

        # Fetch papers from the day before execution
        # arXiv indexes papers overnight — yesterday's papers are complete
        target_dt = execution_dt - timedelta(days=1)
        target_date = target_dt.strftime("%Y%m%d")  # arXiv format: YYYYMMDD

        logger.info(f"Fetching papers for date: {target_date}")

        # asyncio.run() bridges Airflow's sync context to async pipeline
        results = asyncio.run(
            run_paper_ingestion_pipeline(
                target_date=target_date,
                max_results=10,  # start small — increase in production
                process_pdfs=True,
            )
        )

        logger.info(f"Daily fetch complete: {results}")

        # Push results to XCom — downstream tasks will read this
        context["task_instance"].xcom_push(key="fetch_results", value=results)

        return results

    except Exception as e:
        error_msg = f"Daily paper fetch failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def process_failed_pdfs(**context):
    """
    Task 3 — investigate any failures from the main fetch task.
    Runs in parallel with create_opensearch_placeholders.

    Week 2: logs errors for investigation.
    Week 3+: could retry with different settings or smaller page limits.
    """
    logger.info("Processing failed PDFs")

    try:
        # Pull results from the fetch task via XCom
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        if not fetch_results or not fetch_results.get("errors"):
            logger.info("No failures to retry")
            return {"status": "skipped", "message": "No failures"}

        errors = fetch_results["errors"]
        logger.info(f"Found {len(errors)} errors to investigate")

        for error in errors:
            logger.warning(f"Error: {error}")

        return {
            "status": "analysed",
            "errors_logged": len(errors),
            "message": "Errors logged for investigation",
        }

    except Exception as e:
        error_msg = f"Failed PDF processing error: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def create_opensearch_placeholders(**context):
    """
    Task 4 — placeholder for Week 3 OpenSearch indexing.
    Runs in parallel with process_failed_pdfs.

    Week 2: just logs what would be indexed.
    Week 3: actually creates OpenSearch documents.
    """
    logger.info("Creating OpenSearch placeholders (Week 2 stub)")

    try:
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")

        if not fetch_results:
            return {"status": "skipped", "message": "No papers to process"}

        papers_stored = fetch_results.get("papers_stored", 0)
        logger.info(f"Week 2: {papers_stored} papers ready for future OpenSearch indexing")

        return {
            "status": "placeholder",
            "papers_ready_for_indexing": papers_stored,
            "message": f"{papers_stored} papers ready for Week 3 indexing",
        }

    except Exception as e:
        error_msg = f"OpenSearch placeholder failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def generate_daily_report(**context):
    """
    Task 5 — collects results from all upstream tasks and logs a summary.
    Runs after both parallel tasks complete.

    This is your morning dashboard — tells you exactly what happened overnight.
    Like a Spring Batch JobExecutionListener.afterJob().
    """
    logger.info("Generating daily processing report")

    try:
        # Pull results from all upstream tasks
        fetch_results = context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")
        failed_pdf_results = context["task_instance"].xcom_pull(task_ids="process_failed_pdfs")
        opensearch_results = context["task_instance"].xcom_pull(task_ids="create_opensearch_placeholders")

        report = {
            "date": context["ds"],
            "execution_time": datetime.now().isoformat(),
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
                "papers_ready": opensearch_results.get("papers_ready_for_indexing", 0) if opensearch_results else 0,
                "status": opensearch_results.get("status", "unknown") if opensearch_results else "unknown",
            },
        }

        # Print the report to Airflow logs
        logger.info("=" * 50)
        logger.info("DAILY ARXIV PROCESSING REPORT")
        logger.info("=" * 50)
        logger.info(f"Date:             {report['date']}")
        logger.info(f"Papers fetched:   {report['papers']['fetched']}")
        logger.info(f"PDFs downloaded:  {report['papers']['pdfs_downloaded']}")
        logger.info(f"PDFs parsed:      {report['papers']['pdfs_parsed']}")
        logger.info(f"Papers stored:    {report['papers']['stored']}")
        logger.info(f"Processing time:  {report['processing']['time_seconds']:.1f}s")
        logger.info(f"Errors:           {report['processing']['errors']}")
        logger.info(f"OpenSearch ready: {report['opensearch']['papers_ready']}")
        logger.info("=" * 50)

        return report

    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
