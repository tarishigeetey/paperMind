from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    What the caller sends to POST /api/v1/search/
    Like a @RequestBody DTO in Spring.
    """

    query: str = Field(..., min_length=1, max_length=500, description="Search query across title, abstract, and authors")
    size: int = Field(
        default=10,
        ge=1,  # >= 1
        le=50,  # <= 50 — prevent huge result sets
        description="Number of results to return",
    )
    # alias="from" because "from" is a Python reserved keyword
    # JSON: {"from": 0} → Python: from_=0
    from_: int = Field(default=0, ge=0, alias="from", description="Pagination offset")
    categories: Optional[List[str]] = Field(default=None, description="Filter by arXiv categories e.g. ['cs.AI', 'cs.LG']")
    latest_papers: bool = Field(default=False, description="Sort by publication date instead of relevance")


class SearchHit(BaseModel):
    """
    A single search result.
    Like a row in a SQL result set.
    """

    arxiv_id: str
    title: str
    authors: Optional[str]  # joined string from OpenSearch
    abstract: Optional[str]
    published_date: Optional[str]
    pdf_url: Optional[str]
    score: float  # BM25 relevance score
    highlights: Optional[Dict[str, Any]] = None  # matched text snippets


class SearchResponse(BaseModel):
    """
    Full search response.
    Like a Page<T> in Spring Data.
    """

    query: str
    total: int  # total matching documents
    hits: List[SearchHit]  # current page of results
    error: Optional[str] = None  # non-null if something went wrong
