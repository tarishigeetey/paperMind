from functools import lru_cache
from src.config import get_settings
from .parser import PDFParserService


@lru_cache(maxsize=1)
def make_pdf_parser_service() -> PDFParserService:
    """
    Factory function — creates PDFParserService with settings.

    Uses @lru_cache because Docling loads PyTorch models on init.
    Loading models takes 10-30 seconds and uses ~2GB RAM.
    @lru_cache ensures models load ONCE per process, never again.

    Like a @Singleton bean in Spring — one instance, shared everywhere.
    """
    settings = get_settings()
    return PDFParserService(
        max_pages=settings.pdf_parser.max_pages,
        max_file_size_mb=settings.pdf_parser.max_file_size_mb,
        do_ocr=settings.pdf_parser.do_ocr,
        do_table_structure=settings.pdf_parser.do_table_structure,
    )


def reset_pdf_parser() -> None:
    """
    Clear the cached instance.
    Used in tests to get a fresh parser with different settings.
    """
    make_pdf_parser_service.cache_clear()
