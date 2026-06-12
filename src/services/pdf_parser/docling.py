import logging
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import (
    PaperSection,
    ParserType,
    PdfContent,
)

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Low-level Docling PDF parser.
    Handles validation, model warmup, and content extraction.
    Like a low-level @Repository — raw operations, no business logic.
    """

    def __init__(
        self,
        max_pages: int = 20,
        max_file_size_mb: int = 20,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ):
        """
        Initialize DocumentConverter with pipeline options.
        The converter is created once and reused — models are expensive to load.
        Like a singleton JPA EntityManagerFactory.
        """
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,  # OCR is very slow — off by default
        )

        # DocumentConverter is the main Docling entry point
        # We configure it once with our pipeline options
        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _warm_up_models(self):
        """
        Mark models as warmed up.
        On first use Docling loads PyTorch models into memory.
        This flag prevents repeated initialisation checks.
        """
        if not self._warmed_up:
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """
        Validate PDF before expensive parsing.
        Three checks: file size, page count, PDF header.
        Fail fast — don't waste time on invalid files.
        """
        # Check file is not empty
        if pdf_path.stat().st_size == 0:
            raise PDFValidationError(f"PDF file is empty: {pdf_path}")

        # Check file size limit
        file_size = pdf_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise PDFValidationError(
                f"PDF too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB limit"
            )

        # Check PDF magic bytes — first 5 bytes must be "%PDF-"
        # A renamed .pdf file that isn't actually a PDF will fail here
        with open(pdf_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                raise PDFValidationError(f"Not a valid PDF file: {pdf_path}")

        # Check page count using pypdfium2 — faster than loading full Docling
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        actual_pages = len(pdf_doc)
        pdf_doc.close()

        if actual_pages > self.max_pages:
            raise PDFValidationError(f"PDF has {actual_pages} pages > {self.max_pages} page limit")

        return True

    async def parse_pdf(self, pdf_path: Path) -> Optional["PdfContent"]:
        """
        Parse a PDF file into structured content.

        Returns PdfContent with sections, raw_text, tables.
        Returns None if PDF exceeds size/page limits (graceful skip).
        Raises PDFParsingException for genuine errors.
        """
        try:
            # Validate first — fail fast before loading models
            self._validate_pdf(pdf_path)
            self._warm_up_models()

            # Convert PDF — this is where Docling's deep learning runs
            result = self._converter.convert(
                str(pdf_path),
                max_num_pages=self.max_pages,
                max_file_size=self.max_file_size_bytes,
            )

            doc = result.document

            # ── Extract sections ───────────────────────────────────
            # Docling gives us document elements with labels
            # We walk through them building sections as we find headers
            sections = []
            current_section = {"title": "Content", "content": ""}

            for element in doc.texts:
                if hasattr(element, "label") and element.label in ["title", "section_header"]:
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        sections.append(
                            PaperSection(
                                title=current_section["title"],
                                content=current_section["content"].strip(),
                            )
                        )
                    # Start new section
                    current_section = {
                        "title": element.text.strip(),
                        "content": "",
                    }
                else:
                    # Append text to current section
                    if hasattr(element, "text") and element.text:
                        current_section["content"] += element.text + "\n"

            # Don't forget the last section
            if current_section["content"].strip():
                sections.append(
                    PaperSection(
                        title=current_section["title"],
                        content=current_section["content"].strip(),
                    )
                )

            return PdfContent(
                sections=sections,
                figures=[],  # not needed for RAG
                tables=[],  # not needed for Week 2
                raw_text=doc.export_to_text(),  # full text for search
                references=[],
                parser_used=ParserType.DOCLING,
                metadata={
                    "source": "docling",
                    "note": "Content from PDF, metadata from arXiv API",
                },
            )

        except PDFValidationError as e:
            # Size/page limit — graceful skip, not an error
            error_msg = str(e).lower()
            if "too large" in error_msg or "page" in error_msg:
                logger.info(f"Skipping PDF (limits exceeded): {e}")
                return None
            raise  # corrupted file — re-raise

        except Exception as e:
            logger.error(f"Docling parsing failed for {pdf_path.name}: {e}")
            error_msg = str(e).lower()

            if "not valid" in error_msg:
                raise PDFParsingException(f"Corrupted PDF: {pdf_path}")
            elif "timeout" in error_msg:
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg:
                raise PDFParsingException(f"Out of memory: {pdf_path}")
            else:
                raise PDFParsingException(f"Docling failed for {pdf_path.name}: {e}")
