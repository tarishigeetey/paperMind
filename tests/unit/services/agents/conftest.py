"""Shared fixtures for the agentic RAG unit tests.

Java analogy: this is the @TestConfiguration class that provides mock
@Bean overrides (Mockito @MockBean) shared across a whole test package.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.services.agents.context import Context


@pytest.fixture
def mock_opensearch_client():
    """Mock search client — stands in for whatever implements search_unified()
    (OpenSearchClient historically, PgVectorClient today; both share the same
    interface, see src/services/agents/tools.py)."""
    client = Mock()
    client.search_unified = Mock(
        return_value={
            "hits": [
                {
                    "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani, A.",
                    "score": 0.95,
                    "section_name": "Introduction",
                },
                {
                    "chunk_text": "Self-attention allows the model to weigh the importance of different input tokens.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani, A.",
                    "score": 0.90,
                    "section_name": "Methods",
                },
            ]
        }
    )
    return client


@pytest.fixture
def mock_jina_embeddings_client():
    """Mock embeddings client — stands in for JinaEmbeddingsClient."""
    client = Mock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def mock_ollama_client():
    """Mock LLM client — stands in for whatever implements create_llm()
    (OllamaClient historically, GroqClient today; both share the same
    interface, see src/services/agents/context.py)."""
    client = Mock()
    client.create_llm = Mock(return_value=Mock())
    return client


@pytest.fixture
def test_context(mock_opensearch_client, mock_ollama_client, mock_jina_embeddings_client):
    """A Context with Langfuse disabled — nodes skip span creation entirely."""
    return Context(
        ollama_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        langfuse_enabled=False,
        model_name="llama3.2:1b",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )


@pytest.fixture
def sample_human_message():
    return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message():
    return AIMessage(content="Machine learning is a subfield of AI focused on learning from data.")


@pytest.fixture
def sample_tool_message():
    return ToolMessage(
        content="Transformers are neural network architectures based on self-attention mechanisms.",
        tool_call_id="retrieve_1",
        name="retrieve_papers",
    )
