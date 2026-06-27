"""
The retriever tool — wraps OpenSearch as a LangChain "tool" the LLM can call.

=== JAVA COMPARISON: the big picture ===
This file = a Spring @Bean factory method that builds a callable
component, decorated so a framework can discover and invoke it
dynamically:

    @Configuration
    public class ToolConfig {
        @Bean
        public Tool retrievePapersTool(OpenSearchClient opensearch, JinaEmbeddingsClient embeddings) {
            return Tool.builder()
                .name("retrieve_papers")
                .description("Search and return relevant arXiv research papers...")
                .handler(query -> {
                    // the actual search logic
                })
                .build();
        }
    }

WHY DOES THE LLM NEED A "TOOL" AT ALL?
LLMs can only generate TEXT. They can't directly query a database.
A "tool" is a contract: "if you (the LLM) decide you need external
data, output a structured request describing what you want, and I
(the framework) will execute the actual function and feed the result
back to you." This is EXACTLY the same mental model as a REST client
generated from an OpenAPI spec — the LLM doesn't call OpenSearch
directly, it emits a structured "I want to call retrieve_papers with
query=X" request that something else (LangGraph's ToolNode) executes.

=== THE @tool DECORATOR — Python's version of an annotation ===
@tool is a LangChain decorator. Decorators in Python are conceptually
identical to Java ANNOTATIONS that trigger framework behavior — except
where Java annotations are usually just metadata that some other
processor reads (@Component, @Service), Python decorators actually
WRAP the function with new behavior at definition time.

    Java:   @Component  // metadata read by Spring's classpath scanner
            public class RetrievePapersTool { ... }

    Python: @tool        // ACTIVELY wraps the function into a Tool object
            async def retrieve_papers(query: str): ...

When LangChain sees @tool on a function, it inspects:
  - the function NAME → becomes the tool's name ("retrieve_papers")
  - the DOCSTRING → becomes the tool's description (the LLM reads
    this to decide WHEN to call this tool — see the "Use this tool
    when..." bullet list below, that text is NOT just documentation,
    the LLM actually reads it as part of its decision-making!)
  - the TYPE HINTS → becomes the tool's parameter schema (query: str)

This is similar to how Spring reads @RequestMapping + method signature
+ @RequestParam to build its routing table — except here the
"consumer" of this metadata is an LLM's reasoning process, not an
HTTP router.
"""

import logging

# Document = LangChain's standard wrapper for "a piece of retrieved
# content + its metadata" — used everywhere in RAG pipelines.
# Java equivalent: a simple record like
#   public record Document(String pageContent, Map<String, Object> metadata) {}
from langchain_core.documents import Document

# tool = the decorator that converts a plain async function into a
# LangChain Tool object the LLM can invoke
from langchain_core.tools import tool

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


# ============================================================
# create_retriever_tool — a FACTORY FUNCTION (closures, not classes)
# ============================================================
# WHY A FACTORY FUNCTION INSTEAD OF JUST DEFINING THE TOOL DIRECTLY
# AT MODULE LEVEL?
#
# The tool needs access to opensearch_client, embeddings_client, top_k,
# and use_hybrid — but these come from OUR runtime config (Context),
# not from the LLM's tool call arguments. The LLM should only ever
# specify `query` — it has no business knowing about connection
# details or search tuning parameters.
#
# JAVA EQUIVALENT — this is a CLOSURE pattern, which Java emulates via
# either a factory method returning a lambda, or an inner/anonymous class
# capturing final fields:
#
#     public Function<String, CompletableFuture<List<Document>>> createRetrieverTool(
#             OpenSearchClient opensearchClient,
#             JinaEmbeddingsClient embeddingsClient,
#             int topK,
#             boolean useHybrid) {
#
#         // The returned lambda "closes over" (captures) topK, useHybrid, etc.
#         return (String query) -> {
#             // ... uses opensearchClient, embeddingsClient, topK, useHybrid
#             // from the ENCLOSING scope, not from the lambda's own parameters
#         };
#     }
#
# This is Java's effectively-final variable capture in lambdas — Python
# does the same thing via closures, just without the "effectively final"
# restriction (Python closures can even rebind outer variables with
# `nonlocal`, though we don't need that here).
def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving papers

    JAVA EQUIVALENT (full factory method):
        public Tool createRetrieverTool(
                OpenSearchClient opensearchClient,
                JinaEmbeddingsClient embeddingsClient,
                int topK,
                boolean useHybrid) {
            return Tool.builder()
                .name("retrieve_papers")
                .description("Search and return relevant arXiv research papers...")
                .handler(this::retrievePapers)  // captures opensearchClient etc.
                .build();
        }

    WHO CALLS THIS FACTORY?
    factory.py (the next file we touch, in the upcoming wiring episode)
    calls this ONCE per request, passing in the live OpenSearch/Jina
    clients from Context. The returned `retrieve_papers` function is
    then registered into the LangGraph's ToolNode.
    """

    # === THE @tool DECORATOR — defining the ACTUAL tool ===
    #
    # This inner function is defined INSIDE the factory — that's what
    # makes opensearch_client, embeddings_client, top_k, use_hybrid
    # available to it via closure, without the LLM ever seeing them
    # as parameters.
    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata

        *** CRITICAL: THIS DOCSTRING IS NOT JUST DOCUMENTATION ***
        LangChain extracts this exact docstring text and sends it to
        the LLM as the tool's description. The LLM reads "Use this
        tool when..." and decides FOR ITSELF whether the current
        query matches. This is fundamentally different from a Java
        Javadoc comment, which is purely for human readers and has
        ZERO runtime effect. Here, changing this docstring literally
        changes the AI's behavior — it's both documentation AND
        executable specification at once.

        Java has no real analog for this — the closest mental model
        is OpenAPI's `description` field on an endpoint, which SOME
        tools (like AI agents calling your REST API) might read to
        decide when to call it, but a typical Java service never
        "reads its own Javadoc" to make decisions.
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # === STEP 1: Convert the query text into a vector embedding ===
        # Java: float[] queryEmbedding = embeddingsClient.embedQuery(query).get();
        # (Jina's embedding API turns text into a list of floats —
        # a numeric "fingerprint" of the query's meaning, used for
        # vector/semantic search.)
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        # === STEP 2: Search OpenSearch (hybrid = BM25 + vector) ===
        # Java: SearchResponse results = opensearchClient.searchUnified(
        #           query, queryEmbedding, topK, useHybrid);
        logger.debug("Searching OpenSearch")
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        # === STEP 3: Map raw OpenSearch hits → LangChain Document objects ===
        # WHY MAP AT ALL? Same reason you NEVER pass a raw JDBC
        # ResultSet around your Java codebase — opensearch_client
        # returns its own raw hit format (a dict with whatever keys
        # OpenSearch gave us), but the REST of our agent pipeline
        # (grading, generation) expects LangChain's standard Document
        # type. This mapping step is our "anti-corruption layer" —
        # same concept as a Repository mapping a JPA entity into a
        # domain model before handing it to a @Service.
        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        # Python for-loop over raw hits, building Document objects
        # Java: for (Map<String, Object> hit : hits) { documents.add(new Document(...)); }
        for hit in hits:
            doc = Document(
                # page_content = the actual text chunk that got matched
                page_content=hit["chunk_text"],
                # metadata = everything ELSE useful about this hit —
                # Java: a HashMap<String, Object> attached to the Document
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        # === STEP 4: Return — LangChain auto-serializes this back to
        # the LLM as a ToolMessage ===
        # We don't manually build a ToolMessage here — LangGraph's
        # ToolNode (wired up in agentic_rag.py, coming up) handles
        # wrapping this return value into a proper ToolMessage and
        # appending it to AgentState.messages via the add_messages
        # reducer. We just return the raw business result, same as a
        # Spring @Service method returning a domain object and letting
        # a @RestController layer handle the HTTP response wrapping.
        return documents

    # Return the decorated function itself — NOT calling it.
    # Java: return the Tool object you just built, ready to register
    # into your agent's available-tools list.
    return retrieve_papers
