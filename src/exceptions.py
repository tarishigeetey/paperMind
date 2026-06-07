"""Custom exceptions for PaperMind."""


class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Raised when a paper is not found."""


class PaperNotSaved(RepositoryException):
    """Raised when a paper fails to save."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
