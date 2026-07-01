import logging
from pathlib import Path
from typing import Optional

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .docling import DoclingParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """
    Public PDF parsing interface.
    Routes to DoclingParser.
    Like a @Service wrapping a @Repository.

    In future weeks this could route to different parsers
    based on PDF type — academic, scanned, slides etc.
    """

    def __init__(
        self,
        max_pages: int = 20,
        max_file_size_mb: int = 20,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ):
        self.docling_parser = DoclingParser(
            max_pages=max_pages,
            max_file_size_mb=max_file_size_mb,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse a PDF file.
        Returns PdfContent or None if PDF should be skipped.
        Raises PDFParsingException on genuine errors.
        """
        if not pdf_path.exists():
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = await self.docling_parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name} successfully")
            return result

        except (PDFValidationError, PDFParsingException):
            raise  # let caller handle these specifically
        except Exception as e:
            raise PDFParsingException(f"Unexpected error parsing {pdf_path.name}: {e}")
