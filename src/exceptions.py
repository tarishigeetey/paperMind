"""
PaperMind exception hierarchy.
Week 1: basic repository + config exceptions
Week 2: adds arXiv API, PDF parsing, and pipeline exceptions
"""


# ── Repository exceptions ──────────────────────────────────────────
class RepositoryException(Exception):
    """Base exception for all repository errors."""


class PaperNotFound(RepositoryException):
    """Paper not found in database."""


class PaperNotSaved(RepositoryException):
    """Paper failed to save to database."""


# ── Parsing exceptions ─────────────────────────────────────────────
class ParsingException(Exception):
    """Base exception for all parsing errors."""


class PDFParsingException(ParsingException):
    """PDF parsing failed."""


class PDFValidationError(PDFParsingException):
    """PDF failed validation — too large, too many pages, corrupted."""


# ── PDF download exceptions ────────────────────────────────────────
class PDFDownloadException(Exception):
    """PDF download failed."""


class PDFDownloadTimeoutError(PDFDownloadException):
    """PDF download timed out."""


class PDFCacheException(Exception):
    """PDF cache operation failed."""


# ── arXiv API exceptions ───────────────────────────────────────────
class ArxivAPIException(Exception):
    """Base exception for all arXiv API errors."""


class ArxivAPITimeoutError(ArxivAPIException):
    """arXiv API request timed out."""


class ArxivAPIRateLimitError(ArxivAPIException):
    """arXiv API rate limit exceeded."""


class ArxivParseError(ArxivAPIException):
    """Failed to parse arXiv XML response."""


# ── Pipeline exceptions ────────────────────────────────────────────
class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Pipeline execution failed."""


# ── OpenSearch exceptions (placeholder — Week 3) ───────────────────
class OpenSearchException(Exception):
    """Base exception for OpenSearch errors."""


# ── S3 exceptions (Episode 10.1) ───────────────────────────────────
class S3Exception(Exception):
    """Base exception for S3 storage errors."""


class S3UploadError(S3Exception):
    """Failed to upload an object to S3."""


class S3DownloadError(S3Exception):
    """Failed to download an object from S3."""


# ── LLM exceptions (placeholder — Week 5) ─────────────────────────
class LLMException(Exception):
    """Base exception for LLM errors."""


class OllamaException(LLMException):
    """Ollama service error."""


class OllamaConnectionError(OllamaException):
    """Cannot connect to Ollama."""


class OllamaTimeoutError(OllamaException):
    """Ollama service timed out."""


# ── General exceptions ─────────────────────────────────────────────
class ConfigurationError(Exception):
    """Configuration is invalid or missing."""
