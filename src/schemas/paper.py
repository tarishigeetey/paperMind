from datetime import datetime
from typing import List
from uuid import UUID
from pydantic import BaseModel, Field


class PaperBase(BaseModel):
    """
    Shared fields between create and response.
    Like an abstract base DTO.
    """

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="Author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="arXiv categories")
    published_date: datetime = Field(..., description="Published date")
    pdf_url: str = Field(..., description="PDF URL")


class PaperCreate(PaperBase):
    """
    Used when inserting a paper.
    Like a CreateDTO — only the fields the caller provides.
    No id, no created_at — those are generated server-side.
    """

    pass


class PaperResponse(PaperBase):
    """
    Used when returning a paper to the API caller.
    Adds server-generated fields: id, created_at, updated_at.
    Like a ResponseDTO.
    """

    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # allows building from SQLAlchemy model
        # like @JsonProperty reading from @Entity


class PaperSearchResponse(BaseModel):
    """Paginated list of papers."""

    papers: List[PaperResponse]
    total: int
