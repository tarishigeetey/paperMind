from src.config import get_settings
from src.services.arxiv.client import ArxivClient


def make_arxiv_client() -> ArxivClient:
    """
    Factory function — creates ArxivClient with settings.
    Like a Spring @Bean factory method.
    Called once at Airflow task startup.
    """
    settings = get_settings()
    return ArxivClient(settings=settings.arxiv)
