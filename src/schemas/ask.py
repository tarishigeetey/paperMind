from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """What the caller sends when asking a question."""

    question: str = Field(..., description="Question about arXiv papers")


class PaperSource(BaseModel):
    """A paper cited in the answer."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract_preview: str


class AskResponse(BaseModel):
    """The answer + which papers it came from."""

    answer: str
    sources: List[PaperSource]
