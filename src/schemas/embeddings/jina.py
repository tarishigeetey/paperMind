from typing import Dict, List

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """
    Request body for Jina AI embeddings API.
    Like a @RequestBody DTO for an external API call.

    Two task types:
    - retrieval.passage: for indexing document chunks
    - retrieval.query:   for embedding search queries

    Why different tasks? Jina v3 is a late-interaction model —
    it uses different representations for documents vs queries
    to maximise retrieval accuracy. Asymmetric embedding.
    """

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"  # or "retrieval.query"
    dimensions: int = 1024  # 1024 = best accuracy/speed tradeoff
    late_chunking: bool = False  # we handle chunking ourselves
    embedding_type: str = "float"
    input: List[str]  # texts to embed


class JinaEmbeddingResponse(BaseModel):
    """
    Response from Jina AI embeddings API.

    data is a list of dicts, each containing:
    - embedding: List[float] of length 1024
    - index: int (position in input list)
    - object: "embedding"
    """

    model: str
    object: str = "list"
    usage: Dict[str, int]  # {"total_tokens": N, "prompt_tokens": N}
    data: List[Dict]  # [{"embedding": [...], "index": 0}, ...]
