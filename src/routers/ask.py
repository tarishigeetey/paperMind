import logging

from fastapi import APIRouter

from src.schemas.ask import AskRequest, AskResponse, PaperSource

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    RAG question answering endpoint.
    Week 1: returns a mock response so the endpoint exists.
    Week 5: replaced with real LLM + retrieval pipeline.
    """
    logger.info(f"Question received: {request.question}")

    return AskResponse(
        answer=(
            f"You asked: '{request.question}'. "
            "The RAG pipeline will be implemented in Week 5. "
            "For now, the infrastructure is being set up."
        ),
        sources=[
            PaperSource(
                arxiv_id="2401.00001",
                title="Placeholder Paper",
                authors=["Week 5 Author"],
                abstract_preview="This will be a real paper in Week 5...",
            )
        ],
    )
