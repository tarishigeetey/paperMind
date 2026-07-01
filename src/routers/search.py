import logging

from fastapi import APIRouter, HTTPException

from src.dependencies import OpenSearchDep
from src.schemas.api.search import SearchHit, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_papers(
    request: SearchRequest,
    opensearch_client: OpenSearchDep,
) -> SearchResponse:
    """BM25 search — now searches chunk index."""
    try:
        if not opensearch_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Search service unavailable",
            )

        results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=None,  # BM25 only — no embedding
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=False,  # BM25 only
        )

        hits = [
            SearchHit(
                arxiv_id=hit.get("arxiv_id", ""),
                title=hit.get("title", ""),
                authors=hit.get("authors"),
                abstract=hit.get("abstract"),
                published_date=hit.get("published_date"),
                pdf_url=hit.get("pdf_url"),
                score=hit.get("score", 0.0),
                highlights=hit.get("highlights"),
            )
            for hit in results.get("hits", [])
        ]

        return SearchResponse(
            query=request.query,
            total=results.get("total", 0),
            hits=hits,
            error=results.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
