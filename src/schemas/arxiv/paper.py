from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """
    Raw response from arXiv API.
    Just what the API returns — no DB fields.
    Like a raw API response DTO.
    """

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str  # string from API — parsed to datetime before DB storage
    pdf_url: str


class PaperBase(BaseModel):
    """Shared fields between create and response."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime
    pdf_url: str


class PaperCreate(PaperBase):
    """
    Used when inserting/updating a paper.
    Includes optional PDF content fields — None when PDF parsing skipped or failed.
    """

    # PDF content — all optional because PDF parsing can fail
    raw_text: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    references: Optional[List[Dict[str, Any]]] = None

    # PDF processing metadata
    parser_used: Optional[str] = None
    parser_metadata: Optional[Dict[str, Any]] = None
    pdf_processed: Optional[bool] = False
    pdf_processing_date: Optional[datetime] = None


class PaperResponse(PaperBase):
    """
    What the API returns to callers.
    Includes server-generated fields + PDF content if available.
    """

    id: UUID
    raw_text: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    references: Optional[List[Dict[str, Any]]] = None
    parser_used: Optional[str] = None
    parser_metadata: Optional[Dict[str, Any]] = None
    pdf_processed: bool = False
    pdf_processing_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # read from SQLAlchemy model objects


class PaperSearchResponse(BaseModel):
    """Paginated list of papers."""

    papers: List[PaperResponse]
    total: int
