from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParserType(str, Enum):
    """
    Which parser was used.
    str Enum means the value serialises to a plain string in JSON.
    Like an @JsonValue enum in Java.
    """

    DOCLING = "docling"
    GROBID = "grobid"  # future use


class PaperSection(BaseModel):
    """One section of a paper — Introduction, Methods, Results etc."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, description="Nesting level")


class PaperFigure(BaseModel):
    """A figure extracted from the PDF."""

    caption: str
    id: str


class PaperTable(BaseModel):
    """A table extracted from the PDF."""

    caption: str
    id: str


class PdfContent(BaseModel):
    """
    Everything Docling extracted from a PDF.
    This is the output of the parsing step.
    """

    sections: List[PaperSection] = Field(default_factory=list)
    figures: List[PaperFigure] = Field(default_factory=list)
    tables: List[PaperTable] = Field(default_factory=list)
    raw_text: str = Field(..., description="Full extracted text")
    references: List[str] = Field(default_factory=list)
    parser_used: ParserType = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArxivMetadata(BaseModel):
    """
    Paper metadata from the arXiv API.
    Separate from PdfContent — two different sources of truth.
    """

    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    categories: List[str] = Field(default_factory=list)
    published_date: str
    pdf_url: str


class ParsedPaper(BaseModel):
    """
    Complete paper — arXiv metadata + PDF content combined.
    This is what MetadataFetcher produces after the full pipeline.
    Like a composite DTO combining two sources.
    """

    arxiv_metadata: ArxivMetadata
    pdf_content: Optional[PdfContent] = None
