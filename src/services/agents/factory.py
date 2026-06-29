"""
Factory for constructing a fully-wired AgenticRAGService.

=== JAVA COMPARISON ===
This entire file is ONE Spring @Bean factory method:

    @Configuration
    public class AgenticRagConfig {
        @Bean
        public AgenticRAGService agenticRagService(
                OpenSearchClient opensearchClient,
                OllamaClient ollamaClient,
                JinaEmbeddingsClient embeddingsClient,
                @Nullable LangfuseTracer langfuseTracer,
                @Value("${agent.top-k:3}") int topK,
                @Value("${agent.use-hybrid:true}") boolean useHybrid) {

            GraphConfig graphConfig = new GraphConfig();
            graphConfig.setTopK(topK);
            graphConfig.setUseHybrid(useHybrid);

            return new AgenticRAGService(
                opensearchClient, ollamaClient, embeddingsClient,
                langfuseTracer, graphConfig);
        }
    }

WHY HAVE A SEPARATE factory.py WHEN AgenticRAGService's __init__
ALREADY ACCEPTS ALL THESE PARAMETERS DIRECTLY?
This is the FACTORY PATTERN we flagged back in Episode 1 when we
first created this folder structure. The constructor (__init__ in
agentic_rag.py) takes a fully-built GraphConfig object — it doesn't
know or care about simpler primitive params like top_k/use_hybrid.
This factory function is the layer that translates "simple,
commonly-tweaked parameters" (top_k, use_hybrid) into "the fully
assembled GraphConfig object" the constructor actually wants.

This separation matters in dependencies.py (where FastAPI's DI system
calls this factory, not the constructor directly) — it gives you ONE
place to change "what counts as a sensible default" without touching
AgenticRAGService's actual constructor signature. Same reason Spring
teams often prefer a @Bean factory METHOD over relying on Spring to
autowire a constructor directly — it's a seam for translating
config/properties into the exact shape a class's constructor expects.
"""

from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    ollama_client: OllamaClient,
    embeddings_client: JinaEmbeddingsClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    model: str = "llama3.2:1b",
) -> AgenticRAGService:
    """
    Create AgenticRAGService with dependency injection.

    Args:
        opensearch_client: Client for document search
        ollama_client: Client for LLM generation
        embeddings_client: Client for embeddings
        langfuse_tracer: Optional Langfuse tracer for observability
        top_k: Number of documents to retrieve (default: 3)
        use_hybrid: Use hybrid search (default: True)

    Returns:
        Configured AgenticRAGService instance

    JAVA EQUIVALENT (the full @Bean method body):
        public AgenticRAGService makeAgenticRagService(
                OpenSearchClient opensearchClient,
                OllamaClient ollamaClient,
                JinaEmbeddingsClient embeddingsClient,
                @Nullable LangfuseTracer langfuseTracer,
                int topK,
                boolean useHybrid) {

            GraphConfig graphConfig = GraphConfig.builder()
                .topK(topK)
                .useHybrid(useHybrid)
                .build();   // every OTHER GraphConfig field uses its default

            return new AgenticRAGService(
                opensearchClient, ollamaClient, embeddingsClient,
                langfuseTracer, graphConfig);
        }

    NOTICE WHAT THIS FUNCTION DOES NOT EXPOSE: max_retrieval_attempts,
    guardrail_threshold, model, temperature, enable_tracing — all of
    GraphConfig's OTHER fields (config.py, Episode 2) just use their
    class defaults here. This factory deliberately only surfaces the
    TWO knobs (top_k, use_hybrid) that the caller (dependencies.py,
    next episode) is expected to tune per-deployment. Everything else
    stays a code-level default unless someone deliberately widens this
    factory's signature later. This is a real API design choice: don't
    expose every possible knob through every layer "just in case" —
    only thread through what callers actually need to vary.
    """
    # Build the GraphConfig, overriding only top_k and use_hybrid —
    # every other field (model, temperature, guardrail_threshold,
    # max_retrieval_attempts, enable_tracing) takes whatever default
    # was defined on the GraphConfig class itself (config.py, Episode 2).
    #
    # Java: GraphConfig graphConfig = new GraphConfig();
    #       graphConfig.setTopK(topK);
    #       graphConfig.setUseHybrid(useHybrid);
    #       // every other field keeps its class-level default
    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
        model=model,
    )

    # Construct and return the fully-wired service. This is the ONLY
    # place in the whole codebase that should call
    # AgenticRAGService(...) directly — everything else (the FastAPI
    # dependency injection layer, next episode) calls THIS factory
    # function instead, never the raw constructor.
    #
    # Java: return new AgenticRAGService(opensearchClient, ollamaClient,
    #           embeddingsClient, langfuseTracer, graphConfig);
    return AgenticRAGService(
        opensearch_client=opensearch_client,
        ollama_client=ollama_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
