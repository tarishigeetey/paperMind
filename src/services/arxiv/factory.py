from src.config import get_settings
from src.services.arxiv.client import ArxivClient
from src.services.s3.factory import make_s3_client


def make_arxiv_client() -> ArxivClient:
    """
    Factory function — creates ArxivClient with settings.
    Like a Spring @Bean factory method.
    Called once at Airflow task startup.
    """
    settings = get_settings()
    return ArxivClient(settings=settings.arxiv, s3_client=make_s3_client())
