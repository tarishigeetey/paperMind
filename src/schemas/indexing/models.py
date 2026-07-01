from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Technical metadata about a text chunk.
    Answers: where does this chunk come from in the original paper?

    Like a SourcePosition in a compiler — tracks provenance.
    """

    chunk_index: int = Field(..., description="Position of this chunk in the paper (0-based)")
    start_char: int = Field(..., description="Character offset where this chunk starts in original text")
    end_char: int = Field(..., description="Character offset where this chunk ends")
    word_count: int = Field(..., description="Number of words in this chunk")
    overlap_with_previous: int = Field(..., description="Words shared with the previous chunk")
    overlap_with_next: int = Field(..., description="Words shared with the next chunk")
    section_title: Optional[str] = Field(default=None, description="Section this chunk belongs to (e.g. 'Introduction')")


class TextChunk(BaseModel):
    """
    A chunk of text extracted from a paper, ready for embedding and indexing.
    Flows from TextChunker → HybridIndexingService → OpenSearch.
    Like a Message in a message queue — produced by one service,
    consumed by another, with metadata for tracing.
    """

    text: str = Field(..., description="The actual chunk text content")
    metadata: ChunkMetadata = Field(..., description="Technical metadata about this chunk")
    arxiv_id: str = Field(..., description="arXiv paper ID this chunk belongs to")
    paper_id: str = Field(..., description="PostgreSQL UUID of the paper")

    @property
    def chunk_id(self) -> str:
        """
        Unique identifier for this chunk.
        Format: arxiv_id__chunk_N
        Used as the OpenSearch document ID — prevents duplicates on reindex.
        Like a composite primary key in a database.
        """
        return f"{self.arxiv_id}__chunk_{self.metadata.chunk_index}"

    @property
    def word_count(self) -> int:
        """Convenience accessor for word count."""
        return self.metadata.word_count
