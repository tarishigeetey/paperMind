import logging
from typing import List

from fastapi import APIRouter, HTTPException

from src.dependencies import SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate, PaperResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("/", response_model=PaperResponse, status_code=201)
def create_paper(paper: PaperCreate, db: SessionDep) -> PaperResponse:
    """
    Insert a new paper.
    Week 2 — Airflow will call this automatically.
    Week 1 — you can test it manually via Swagger UI.
    """
    try:
        repo = PaperRepository(db)
        created = repo.upsert(paper)
        return PaperResponse.model_validate(created)
    except Exception as e:
        logger.error(f"Failed to create paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{arxiv_id}", response_model=PaperResponse)
def get_paper(arxiv_id: str, db: SessionDep) -> PaperResponse:
    """Fetch a single paper by arXiv ID."""
    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    return PaperResponse.model_validate(paper)


@router.get("/", response_model=List[PaperResponse])
def list_papers(
    db: SessionDep,
    limit: int = 20,
    offset: int = 0,
) -> List[PaperResponse]:
    """
    Paginated list of papers, newest first.
    limit/offset come from query params automatically:
    GET /papers?limit=10&offset=20
    Like @RequestParam in Spring.
    """
    repo = PaperRepository(db)
    papers = repo.get_all(limit=limit, offset=offset)
    return [PaperResponse.model_validate(p) for p in papers]
