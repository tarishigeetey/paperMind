import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import EmbeddingsDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
):
    """Shared function to prepare chunks and sources for RAG."""
    query_embedding = None
    search_mode = "bm25"

    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            search_mode = "hybrid"
            logger.info("Generated query embedding for hybrid search")
        except Exception as e:
            logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
            query_embedding = None
            search_mode = "bm25"

    logger.info(f"Retrieving top {request.top_k} chunks for query: '{request.query}'")

    search_results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    chunks = []
    sources_set = set()

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")

        chunk_data = {
            "arxiv_id": arxiv_id,
            "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
        }
        chunks.append(chunk_data)

        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            sources_set.add(pdf_url)

    sources = list(sources_set)

    return chunks, sources, search_mode


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
) -> AskResponse:
    """
    RAG endpoint for question answering.

    1. Retrieves top-k chunks using hybrid search
    2. Passes chunks to LLM with prompt template
    3. Returns structured answer with sources
    """
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        try:
            await ollama_client.health_check()
        except Exception as e:
            logger.error(f"Ollama service unavailable: {e}")
            raise HTTPException(status_code=503, detail="LLM service is currently unavailable")

        chunks, sources, search_mode = await _prepare_chunks_and_sources(request, opensearch_client, embeddings_service)

        if not chunks:
            logger.warning(f"No chunks found for query: {request.query}")
            return AskResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the papers to answer your question.",
                sources=[],
                chunks_used=0,
                search_mode=search_mode,
            )

        logger.info(f"Retrieved {len(chunks)} chunks, generating answer with {request.model}")

        rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)

        logger.debug(f"RAG response: {rag_response}")

        response = AskResponse(
            query=request.query,
            answer=rag_response.get("answer", "Unable to generate answer"),
            sources=sources,
            chunks_used=len(chunks),
            search_mode=search_mode,
        )

        logger.info(f"Successfully generated answer for query: '{request.query}'")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
) -> StreamingResponse:
    """Streaming RAG endpoint - returns answer as it's generated."""

    async def generate_stream():
        try:
            if not opensearch_client.health_check():
                yield f"data: {json.dumps({'error': 'Search service unavailable'})}\n\n"
                return

            await ollama_client.health_check()

            chunks, sources, search_mode = await _prepare_chunks_and_sources(request, opensearch_client, embeddings_service)

            if not chunks:
                yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'sources': sources, 'chunks_used': len(chunks), 'search_mode': search_mode})}\n\n"

            full_response = ""
            async for chunk in ollama_client.generate_rag_answer_stream(query=request.query, chunks=chunks, model=request.model):
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                if chunk.get("done", False):
                    yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )