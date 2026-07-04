# Java analogy: @Bean factory method in Spring — creates a singleton
# and caches it. Identical pattern to src/services/groq/factory.py.

from functools import lru_cache

from src.config import get_settings
from src.services.s3.client import S3Client


@lru_cache(maxsize=1)
def make_s3_client() -> S3Client:
    """
    Create and return a singleton S3 client instance.
    Java analogy: @Bean with @Scope("singleton") in Spring context.
    """
    settings = get_settings()
    return S3Client(settings)
