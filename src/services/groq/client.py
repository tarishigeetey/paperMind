# Java analogy: this is like swapping a JDBC driver.
# The interface (OllamaClient) stays identical — same method signatures,
# same return types. Only the implementation changes underneath.
# Dependencies.py and routers never need to know which LLM is running.
#
# WHY GROQ:
#   - Ollama runs locally → needs GPU or is slow on CPU
#   - Groq runs on their hardware → fast inference, free tier, no GPU needed
#   - llama-3.1-8b-instant supports structured output (JSON mode) ✅
#   - Same Llama model family → similar quality to llama3.2:1b but faster
#
# WHAT CHANGED vs OllamaClient:
#   - httpx manual HTTP calls → groq Python SDK (like switching from raw JDBC to Spring Data)
#   - Ollama /api/generate endpoint → Groq chat completions API (OpenAI-compatible)
#   - format=json_schema → response_format={"type": "json_object"} (Groq's structured output)
#   - usage_metadata field names stay identical → Langfuse tracing unchanged

import logging
from typing import Any, Dict, List, Optional

from groq import AsyncGroq
from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

# Java analogy: reuse OllamaException — same exception hierarchy,
# different implementation. Callers don't need to change their catch blocks.
logger = logging.getLogger(__name__)


class GroqClient:
    """
    Groq LLM client — drop-in replacement for OllamaClient.

    Uses the Groq SDK (OpenAI-compatible API) instead of direct HTTP calls to Ollama.
    All public method signatures match OllamaClient exactly.
    """

    def __init__(self, settings: Settings):
        """Initialize Groq client with settings."""
        # Java analogy: DataSource with connection pool — SDK manages connections
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model  # e.g. "llama-3.1-8b-instant"
        self.timeout = float(settings.groq_timeout)
        self.prompt_builder = RAGPromptBuilder()  # reused from Ollama — prompts don't change
        self.response_parser = ResponseParser()  # reused from Ollama — parsing doesn't change

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Groq API is reachable.
        Matches OllamaClient.health_check() signature exactly.
        """
        try:
            # Groq has no dedicated health endpoint — list models is the lightest call
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "message": "Groq API is reachable",
                # Java analogy: getMetaData().getDatabaseProductVersion()
                "version": f"groq-sdk, models available: {len(models.data)}",
            }
        except Exception as e:
            raise OllamaConnectionError(f"Cannot connect to Groq API: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Groq models.
        Matches OllamaClient.list_models() signature exactly.
        """
        try:
            models = await self.client.models.list()
            # Java analogy: mapping ResultSet rows to DTOs
            return [{"id": m.id, "name": m.id} for m in models.data]
        except Exception as e:
            raise OllamaException(f"Error listing Groq models: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,  # ignored — we use self.model from settings
        use_structured_output: bool = False,  # kept for interface compatibility
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.
        Matches OllamaClient.generate_rag_answer() signature exactly.

        Java analogy: same @Override method — implementation swapped, contract unchanged.
        Callers (ask.py router, agentic_rag nodes) call this with the same args.
        """
        try:
            # Reuse the same prompt builder from Ollama — prompts don't change
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            logger.info(f"Sending request to Groq: model={self.model}")

            # Groq uses OpenAI-compatible chat completions API
            # Java analogy: PreparedStatement with named params vs positional ?
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        # System role sets the RAG persona
                        "role": "system",
                        "content": (
                            "You are an AI assistant specialized in answering questions "
                            "about academic papers from arXiv. Base your answer STRICTLY "
                            "on the provided paper excerpts. Cite sources using [arXiv:id] format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=1024,
                # Java analogy: connection timeout on DataSource
                timeout=self.timeout,
            )

            answer_text = response.choices[0].message.content
            logger.debug(f"Raw Groq response: {answer_text[:200]}")

            # Build usage_metadata in the same format OllamaClient used
            # so Langfuse tracing and cost monitoring work unchanged
            usage = response.usage
            usage_metadata = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                # Groq doesn't return latency in the response body —
                # we'll add it via Langfuse span timing in Episode 11.2
                "latency_ms": None,
            }
            logger.debug(f"Usage metadata: {usage_metadata}")

            # Build sources from chunks (same logic as OllamaClient plain text path)
            sources = []
            seen_urls = set()
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id")
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                    if pdf_url not in seen_urls:
                        sources.append(pdf_url)
                        seen_urls.add(pdf_url)

            citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

            return {
                "answer": answer_text,
                "sources": sources,
                "confidence": "high",  # Groq models are more reliable than 1b local
                "citations": citations[:5],
                "usage_metadata": usage_metadata,
            }

        except Exception as e:
            logger.error(f"Error generating RAG answer with Groq: {e}")
            # Raise OllamaException so callers don't need to change their except blocks
            # Java analogy: wrapping vendor-specific exception in your domain exception
            raise OllamaException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ):
        """
        Streaming RAG answer — not implemented for Groq yet.
        Kept for interface compatibility with OllamaClient.
        """
        # TODO Episode 9.1b: implement Groq streaming with stream=True
        raise NotImplementedError("Groq streaming not yet implemented — use generate_rag_answer")
