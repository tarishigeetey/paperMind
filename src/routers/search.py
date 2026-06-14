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
    """
    Search papers using BM25 scoring in OpenSearch.

    Searches across:
    - title (3x boost)
    - abstract (2x boost)
    - authors (1x boost)

    Results ranked by BM25 relevance score by default.
    Set latest_papers=true to sort by publication date instead.
    """
    try:
        # Health check first — return 503 if search is unavailable
        # Better than letting the search fail with a cryptic error
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        logger.info(f"Search: '{request.query}' (size={request.size}, latest={request.latest_papers})")

        results = opensearch_client.search_papers(
            query=request.query,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest_papers=request.latest_papers,
        )

        # Convert raw dicts to typed SearchHit objects
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
        raise  # re-raise our own HTTP exceptions unchanged
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
