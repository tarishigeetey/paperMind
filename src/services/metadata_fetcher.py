import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """
    Orchestrates the complete arXiv paper ingestion pipeline.

    Three steps:
    1. Fetch metadata from arXiv API
    2. Download and parse PDFs concurrently
    3. Store everything to PostgreSQL

    Like a Spring Batch Job with three Steps.
    """

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
    ):
        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing

    async def fetch_and_process_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline — fetch, process, store.

        Returns a results dict with statistics:
        {
            "papers_fetched": 10,
            "pdfs_downloaded": 9,
            "pdfs_parsed": 8,
            "papers_stored": 10,
            "errors": [...],
            "processing_time": 142.3
        }
        """
        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # ── Step 1: Fetch paper metadata ──────────────────────
            logger.info("Step 1: Fetching papers from arXiv API...")
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results,
                from_date=from_date,
                to_date=to_date,
                sort_by="submittedDate",
                sort_order="descending",
            )

            results["papers_fetched"] = len(papers)
            logger.info(f"Fetched {len(papers)} papers from arXiv")

            if not papers:
                logger.warning("No papers found for this date range")
                return results

            # ── Step 2: Download and parse PDFs ───────────────────
            pdf_results = {}
            if process_pdfs:
                logger.info("Step 2: Processing PDFs...")
                pdf_results = await self._process_pdfs_batch(papers)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            # ── Step 3: Store to database ─────────────────────────
            if store_to_db and db_session:
                logger.info("Step 3: Storing papers to database...")
                stored_count = self._store_papers_to_db(
                    papers,
                    pdf_results.get("parsed_papers", {}),
                    db_session,
                )
                results["papers_stored"] = stored_count
            elif store_to_db:
                logger.warning("Storage requested but no DB session provided")
                results["errors"].append("No database session provided")

            # ── Calculate timing ───────────────────────────────────
            results["processing_time"] = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Pipeline complete in {results['processing_time']:.1f}s: "
                f"{results['papers_fetched']} fetched, "
                f"{results['pdfs_downloaded']} downloaded, "
                f"{results['papers_stored']} stored, "
                f"{len(results['errors'])} errors"
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(
        self,
        papers: List[ArxivPaper],
    ) -> Dict[str, Any]:
        """
        Process all PDFs concurrently using asyncio.Semaphore.

        Two semaphores control concurrency:
        - download_semaphore: max concurrent downloads (I/O bound)
        - parse_semaphore: max concurrent parsers (CPU/RAM bound)

        Each paper goes through its own download+parse pipeline.
        asyncio.gather runs all pipelines concurrently.
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(
            f"Processing {len(papers)} PDFs "
            f"({self.max_concurrent_downloads} concurrent downloads, "
            f"{self.max_concurrent_parsing} concurrent parsers)"
        )

        # Semaphores limit concurrency without blocking the event loop
        # Like a ThreadPoolExecutor but for async code
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Create one pipeline coroutine per paper
        # gather() runs them all concurrently
        # return_exceptions=True means one failure doesn't cancel others
        pipeline_tasks = [self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore) for paper in papers]

        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, BaseException):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {result}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            elif result:
                download_success, parsed_paper = result

                if download_success:
                    results["downloaded"] += 1

                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    results["download_failures"].append(paper.arxiv_id)

        logger.info(f"PDF processing complete: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")

        return results

    async def _download_and_parse_pipeline(
        self,
        paper: ArxivPaper,
        download_semaphore: asyncio.Semaphore,
        parse_semaphore: asyncio.Semaphore,
    ) -> tuple:
        """
        Single paper pipeline — download then parse.

        The semaphores ensure:
        - Max 5 papers downloading simultaneously
        - Max 3 papers being parsed simultaneously
        - Other papers wait their turn without blocking

        Returns: (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        try:
            # ── Download ───────────────────────────────────────────
            # async with semaphore: acquire slot, run, release slot
            # Like a permits-based rate limiter
            async with download_semaphore:
                logger.debug(f"Downloading: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper)

                if not pdf_path:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # ── Parse ──────────────────────────────────────────────
            # Note: parse happens OUTSIDE download_semaphore
            # So other downloads can proceed while this PDF is parsing
            async with parse_semaphore:
                logger.debug(f"Parsing: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                if pdf_content:
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )
                    parsed_paper = ParsedPaper(
                        arxiv_metadata=arxiv_metadata,
                        pdf_content=pdf_content,
                    )
                    logger.debug(f"Parsed {paper.arxiv_id}: {len(pdf_content.raw_text)} chars extracted")
                    return (True, parsed_paper)
                else:
                    # PDF skipped (too large/many pages) — not an error
                    logger.warning(f"PDF skipped for {paper.arxiv_id} (size/page limits)")
                    return (True, None)

        except Exception as e:
            raise MetadataFetchingException(f"Pipeline error for {paper.arxiv_id}: {e}") from e

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> Dict[str, Any]:
        """
        Convert ParsedPaper into a dict ready for database storage.
        Pydantic models → plain Python dicts → SQLAlchemy columns.
        """
        try:
            pdf_content = parsed_paper.pdf_content
            if not pdf_content:
                return {"pdf_processed": False, "parser_metadata": {"error": "No PDF content"}}

            sections = [{"title": s.title, "content": s.content} for s in pdf_content.sections]

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": list(pdf_content.references),
                "parser_used": (pdf_content.parser_used.value if pdf_content.parser_used else None),
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: List[ArxivPaper],
        parsed_papers: Dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """
        Store all papers to PostgreSQL.
        Uses upsert — safe to run multiple times with same papers.
        Commits once at the end for efficiency.
        """
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            try:
                parsed_paper = parsed_papers.get(paper.arxiv_id)

                # Parse the date string from arXiv API to datetime
                # dateutil handles any ISO format arXiv might return
                published_date = (
                    date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
                )

                # Base paper data from arXiv API
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Enrich with PDF content if available
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                else:
                    # Metadata only — PDF parsing skipped or failed
                    paper_data.update(
                        {
                            "pdf_processed": False,
                            "parser_metadata": {"note": "PDF not processed"},
                        }
                    )

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Single commit for all papers — more efficient than per-paper commits
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database")
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    pdf_cache_dir: Optional[Path] = None,
) -> MetadataFetcher:
    """
    Factory for MetadataFetcher.
    Configured for production workloads — 100 papers/day.
    5 concurrent downloads, 1 concurrent parser (Docling is memory intensive).
    """
    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=5,
        max_concurrent_parsing=1,  # Docling uses lots of RAM — keep at 1
    )
