# Java analogy: @Bean factory method in Spring — creates a singleton
# and caches it. Identical pattern to src/services/ollama/factory.py.

from functools import lru_cache

from src.config import get_settings
from src.services.groq.client import GroqClient


@lru_cache(maxsize=1)
def make_groq_client() -> GroqClient:
    """
    Create and return a singleton Groq client instance.
    Java analogy: @Bean with @Scope("singleton") in Spring context.
    """
    settings = get_settings()
    return GroqClient(settings)
