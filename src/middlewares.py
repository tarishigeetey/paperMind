import logging

logger = logging.getLogger(__name__)


def log_request(method: str, path: str) -> None:
    """Log incoming requests."""
    logger.info(f"{method} {path}")


def log_error(error: str, method: str, path: str) -> None:
    """Log errors with context."""
    logger.error(f"Error in {method} {path}: {error}")
