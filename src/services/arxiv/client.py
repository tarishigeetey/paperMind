import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode

import httpx

from src.config import ArxivSettings
from src.exceptions import (
    ArxivAPIException,
    ArxivAPITimeoutError,
    ArxivParseError,
    PDFDownloadException,
    PDFDownloadTimeoutError,
)
from src.schemas.arxiv.paper import ArxivPaper
from src.services.s3.client import S3Client

logger = logging.getLogger(__name__)


class ArxivClient:
    """
    Client for fetching papers from the arXiv API.
    Like a Spring @Service wrapping a WebClient.
    """

    def __init__(self, settings: ArxivSettings, s3_client: Optional[S3Client] = None):
        self._settings = settings
        # Track last request time for rate limiting
        self._last_request_time: Optional[float] = None

        # Episode 10.1: optional durable PDF storage. None (the default)
        # means "local disk only" — same behavior as before this episode,
        # which is why every existing test (constructing ArxivClient with
        # just settings) still works unchanged.
        self._s3_client = s3_client

    @cached_property
    def pdf_cache_dir(self) -> Path:
        """
        PDF cache directory — created lazily on first access.
        @cached_property runs once, then returns the cached value.
        Like @Lazy in Spring — don't create the directory until needed.
        """
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    # ── Simple property accessors ──────────────────────────────────
    # Expose settings as properties so callers don't need to know
    # the settings object exists — encapsulation

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def namespaces(self) -> dict:
        return self._settings.namespaces

    @property
    def rate_limit_delay(self) -> float:
        return self._settings.rate_limit_delay

    @property
    def timeout_seconds(self) -> int:
        return self._settings.timeout_seconds

    @property
    def max_results(self) -> int:
        return self._settings.max_results

    @property
    def search_category(self) -> str:
        return self._settings.search_category

    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[ArxivPaper]:
        """
        Fetch papers from arXiv for the configured category.

        Args:
            max_results: how many papers (default from settings)
            start: pagination offset
            sort_by: submittedDate, lastUpdatedDate, relevance
            sort_order: ascending or descending
            from_date: filter from date (YYYYMMDD format)
            to_date: filter to date (YYYYMMDD format)
        """
        if max_results is None:
            max_results = self.max_results

        # Build the search query
        # cat:cs.AI means "papers in the cs.AI category"
        search_query = f"cat:{self.search_category}"

        # Add date range if provided
        # arXiv date format: YYYYMMDDHHMM
        if from_date or to_date:
            date_from = f"{from_date}0000" if from_date else "*"
            date_to = f"{to_date}2359" if to_date else "*"
            search_query += f" AND submittedDate:[{date_from}+TO+{date_to}]"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),  # arXiv hard limit is 2000
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # safe= tells urlencode NOT to percent-encode these chars
        # arXiv query syntax needs :, +, [, ] to stay as-is
        safe = ":+[]"
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            logger.info(f"Fetching {max_results} {self.search_category} papers from arXiv")

            # Rate limiting — arXiv asks for 3 seconds between requests
            # Without this they will block your IP
            await self._respect_rate_limit()

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()  # raises on 4xx/5xx
                xml_data = response.text

            papers = self._parse_response(xml_data)
            logger.info(f"Fetched {len(papers)} papers")
            return papers

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout: {e}")
            raise ArxivAPITimeoutError(f"arXiv API timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            raise ArxivAPIException(f"arXiv API returned {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch papers: {e}")
            raise ArxivAPIException(f"Unexpected error fetching papers: {e}")

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a single paper by its arXiv ID.
        Used for manual lookups and testing.
        """
        # Strip version suffix if present — "2401.00001v2" → "2401.00001"
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        params = {"id_list": clean_id, "max_results": 1}
        safe = ":+[]*"
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)
            return papers[0] if papers else None

        except httpx.TimeoutException as e:
            raise ArxivAPITimeoutError(f"Timeout fetching paper {arxiv_id}: {e}")
        except httpx.HTTPStatusError as e:
            raise ArxivAPIException(f"HTTP error fetching paper {arxiv_id}: {e}")
        except Exception as e:
            raise ArxivAPIException(f"Error fetching paper {arxiv_id}: {e}")

    async def download_pdf(self, paper: ArxivPaper, force_download: bool = False) -> Optional[Path]:
        """
        Download PDF for a paper to local cache, with S3 as durable backing storage.

        Lookup order (Episode 10.1): local disk -> S3 -> arXiv itself. Local
        disk is a disposable scratch cache (lost on every redeploy); S3 is
        the durable source of truth once a paper has been fetched at least
        once, so a redeploy doesn't mean re-downloading everything from arXiv.

        Returns the path to the downloaded file, or None if failed.
        """
        if not paper.pdf_url:
            logger.error(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        pdf_path = self._get_pdf_path(paper.arxiv_id)

        # Return cached PDF if it exists
        # Like checking a local cache before hitting an API
        if pdf_path.exists() and not force_download:
            logger.info(f"Using cached PDF: {pdf_path.name}")
            return pdf_path

        s3_key = self._get_s3_key(paper.arxiv_id)

        # Second cache tier: S3. Only worth checking if we're not forcing
        # a fresh download and an S3 client was actually configured.
        if self._s3_client is not None and not force_download:
            try:
                if self._s3_client.object_exists(s3_key):
                    self._s3_client.download_file(s3_key, pdf_path)
                    logger.info(f"Restored PDF from S3: {pdf_path.name}")
                    return pdf_path
            except Exception as e:
                # S3 being unavailable shouldn't block us from just
                # downloading from arXiv instead — degrade, don't fail.
                logger.warning(f"S3 lookup failed for {s3_key}, falling back to arXiv: {e}")

        if not await self._download_with_retry(paper.pdf_url, pdf_path):
            return None

        if self._s3_client is not None:
            try:
                self._s3_client.upload_file(pdf_path, s3_key)
            except Exception as e:
                # Same reasoning: a failed upload shouldn't fail the whole
                # download — the caller still has a valid local pdf_path to
                # parse, we just won't have a durable copy this time.
                logger.warning(f"Failed to upload {pdf_path.name} to S3, continuing without it: {e}")

        return pdf_path

    # ── Private helper methods ─────────────────────────────────────

    async def _respect_rate_limit(self) -> None:
        """
        Ensure 3 seconds between requests as arXiv requires.
        Calculates how long since last request and sleeps the difference.
        """
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """Build the local file path for a paper's PDF."""
        # Replace / with _ to make a valid filename
        # "2401.00001" → "2401.00001.pdf"
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return self.pdf_cache_dir / safe_filename

    def _get_s3_key(self, arxiv_id: str) -> str:
        """Build the S3 key for a paper's PDF — same filename, under a pdfs/ prefix."""
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return f"pdfs/{safe_filename}"

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """
        Parse arXiv API XML response into ArxivPaper objects.
        The XML uses Atom format with custom arXiv namespace extensions.
        """
        try:
            root = ET.fromstring(xml_data)

            # findall with namespace — arXiv XML uses Atom namespace
            # "atom:entry" means find all <entry> elements in the atom namespace
            entries = root.findall("atom:entry", self.namespaces)

            papers = []
            for entry in entries:
                paper = self._parse_single_entry(entry)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            raise ArxivParseError(f"Failed to parse arXiv XML: {e}")
        except Exception as e:
            raise ArxivParseError(f"Unexpected error parsing response: {e}")

    def _parse_single_entry(self, entry: ET.Element) -> Optional[ArxivPaper]:
        """Parse one <entry> element into an ArxivPaper object."""
        try:
            arxiv_id = self._get_arxiv_id(entry)
            if not arxiv_id:
                return None

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=self._get_text(entry, "atom:title", clean_newlines=True),
                authors=self._get_authors(entry),
                abstract=self._get_text(entry, "atom:summary", clean_newlines=True),
                published_date=self._get_text(entry, "atom:published"),
                categories=self._get_categories(entry),
                pdf_url=self._get_pdf_url(entry),
            )
        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None  # skip broken entries, don't crash the whole batch

    def _get_text(self, element: ET.Element, path: str, clean_newlines: bool = False) -> str:
        """Safely extract text from an XML element."""
        elem = element.find(path, self.namespaces)
        if elem is None or elem.text is None:
            return ""
        text = elem.text.strip()
        # Abstracts and titles have newlines from XML formatting
        # Replace them with spaces for clean storage
        return text.replace("\n", " ") if clean_newlines else text

    def _get_arxiv_id(self, entry: ET.Element) -> Optional[str]:
        """
        Extract arXiv ID from the entry's <id> element.
        Raw value: "http://arxiv.org/abs/2401.00001v1"
        We want:   "2401.00001v1"
        """
        id_elem = entry.find("atom:id", self.namespaces)
        if id_elem is None or id_elem.text is None:
            return None
        # Split on / and take the last part
        return id_elem.text.split("/")[-1]

    def _get_authors(self, entry: ET.Element) -> List[str]:
        """Extract all author names from the entry."""
        authors = []
        for author in entry.findall("atom:author", self.namespaces):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)
        return authors

    def _get_categories(self, entry: ET.Element) -> List[str]:
        """Extract category terms — e.g. ["cs.AI", "cs.LG"]"""
        categories = []
        for category in entry.findall("atom:category", self.namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _get_pdf_url(self, entry: ET.Element) -> str:
        """
        Find the PDF link in the entry's <link> elements.
        arXiv entries have multiple links — HTML page, PDF, source.
        We want the one with type="application/pdf".
        """
        for link in entry.findall("atom:link", self.namespaces):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                # arXiv sometimes returns http:// — always use https://
                if url.startswith("http://arxiv.org/"):
                    url = url.replace("http://arxiv.org/", "https://arxiv.org/")
                return url
        return ""

    async def _download_with_retry(self, url: str, path: Path, max_retries: int = 3) -> bool:
        """
        Download a file with exponential backoff retry.
        Production pattern — transient failures shouldn't kill the pipeline.

        Attempts: 1st immediately, 2nd after 5s, 3rd after 10s.
        Like Spring Retry's @Retryable with exponential backoff.
        """
        logger.info(f"Downloading PDF from {url}")
        await asyncio.sleep(self.rate_limit_delay)  # respect rate limit

        for attempt in range(max_retries):
            try:
                # stream=True means download in chunks
                # Without streaming, a 20MB PDF loads entirely into memory first
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)

                logger.info(f"Downloaded: {path.name}")
                return True

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise PDFDownloadTimeoutError(f"PDF download timed out after {max_retries} attempts")

            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"HTTP error attempt {attempt + 1}/{max_retries}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise PDFDownloadException(f"PDF download failed after {max_retries} attempts: {e}")

            except Exception as e:
                raise PDFDownloadException(f"Unexpected download error: {e}")

        # Clean up partial download if all retries failed
        if path.exists():
            path.unlink()
        return False
