"""
The Retrieve Node — requests a search, but doesn't execute it directly.

=== JAVA COMPARISON: the big picture ===
This is the MOST conceptually different node so far. Every previous
node (guardrail, out_of_scope) did its work and returned a result
DIRECTLY. This node does something subtler: it builds a TOOL CALL
REQUEST and hands it off — the actual OpenSearch execution happens in
a SEPARATE LangGraph component (a built-in "ToolNode") that we wire up
next episode in agentic_rag.py.

JAVA EQUIVALENT — this is the COMMAND PATTERN:
    public interface Command {
        void execute();
    }

    public class RetrievePapersCommand implements Command {
        private final String query;
        private final String callId;

        public void execute() {
            // actual OpenSearch call happens HERE, in a different class
        }
    }

    // retrieve_node's job: just CREATE the command object
    Command cmd = new RetrievePapersCommand(query, callId);
    commandQueue.add(cmd);   // ← hands off, doesn't call cmd.execute() itself

    // SEPARATELY, a CommandExecutor (= LangGraph's ToolNode) picks
    // commands off the queue and actually runs them

WHY SPLIT "DECIDE TO SEARCH" FROM "ACTUALLY SEARCH" INTO TWO NODES?
1. LangGraph's built-in ToolNode handles ALL tool execution uniformly
   — error handling, retries, response formatting — for EVERY tool in
   your graph, not just retrieve_papers. If you add a second tool
   later (e.g., "summarize_paper"), you don't rewrite this node.
2. It makes the conversation history (state.messages) look EXACTLY
   like a standard LangChain tool-calling conversation: AIMessage
   (with tool_calls) → ToolMessage (the result). This is the same
   shape OpenAI/Anthropic's function-calling APIs produce natively,
   so all the surrounding tooling (Langfuse tracing, message
   filtering in utils.py) works without special-casing our retrieval.

This is the Java equivalent of separating "build the SQL query" from
"execute the SQL query via the connection pool" — the second part is
infrastructure you want to be UNIFORM and REUSABLE, not duplicated in
every node.
"""

import logging
import time

# Dict, Union = Java's Map<K,V> and Java's planned sealed-type union
from typing import Dict, Union

# AIMessage = the message type representing "the assistant decided X"
# Here, AIMessage carries `tool_calls` — a list of structured requests
# the LLM (or in this case, OUR CODE on the LLM's behalf) wants executed
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Initiate retrieval or return fallback if max attempts reached.

    This node creates a tool call to retrieve documents, or returns a fallback
    message if the maximum number of retrieval attempts has been reached.

    :param state: Current agent state
    :param runtime: Runtime context containing max_retrieval_attempts
    :returns: Dictionary with updated state (retrieval_attempts, messages, original_query)

    NOTICE THE RETURN TYPE: Dict[str, Union[int, str, list]]
    Unlike guardrail_node (always returns {"guardrail_result": ...}),
    THIS node's return shape VARIES depending on which branch executes:
      - max attempts reached → {"messages": [...], maybe "original_query": str}
      - normal case          → {"retrieval_attempts": int, "messages": [...], maybe "original_query": str}

    Java equivalent: a method returning different subsets of a DTO's
    fields depending on internal branching — in Java you'd usually
    model this more rigidly (e.g., always populate all fields, use
    null for "not set"), but LangGraph's partial-update model means
    Python can be looser here: only return the keys that actually changed.
    """
    logger.info("NODE: retrieve")
    start_time = time.time()

    # === STEP 1: Read from state — what's the question, how many
    # attempts have we made so far? ===
    messages = state["messages"]
    question = get_latest_query(messages)

    # state.get("retrieval_attempts", 0) = Java's
    #   state.getOrDefault("retrievalAttempts", 0)
    # Using .get() with a default instead of direct [] access — this
    # node might run on the VERY FIRST pass before retrieval_attempts
    # has ever been set, so we need a safe fallback to 0.
    current_attempts = state.get("retrieval_attempts", 0)

    # Java: int maxAttempts = runtime.getContext().getMaxRetrievalAttempts();
    max_attempts = runtime.context.max_retrieval_attempts

    # === STEP 2: Remember the user's ORIGINAL question ===
    # WHY DOES THIS MATTER? After a rewrite_query_node runs (next
    # episode), state["messages"]'s latest HumanMessage might no
    # longer be the user's literal words — it could be the REWRITTEN
    # query. We need to keep the ORIGINAL around so the final API
    # response can show: "You asked: 'tell me about that thing' →
    # We searched: 'attention mechanism transformer architecture'"
    #
    # `updates = {}` here is our PARTIAL UPDATE DICT — we build it up
    # piece by piece and return it at the end. Same pattern as
    # building a Map<String, Object> patch object in Java before
    # returning/merging it.
    updates = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Stored original query: {question[:100]}...")

    # === STEP 3: Langfuse observability span (same pattern as guardrail_node) ===
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_retrieval_initiation",
                input_data={
                    "query": question,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                metadata={
                    "node": "retrieve",
                    "top_k": runtime.context.top_k,
                },
            )
            logger.debug(f"Created Langfuse span for retrieval attempt {current_attempts + 1}")
        except Exception as e:
            logger.warning(f"Failed to create span for retrieve node: {e}")

    # ============================================================
    # BRANCH A: We've already tried the max number of times — GIVE UP
    # ============================================================
    # WHY CHECK THIS HERE, BEFORE SEARCHING AGAIN?
    # This node can be visited MULTIPLE times in one request — recall
    # the cycle: retrieve → grade → (if bad) → rewrite_query → retrieve
    # → grade → ... LangGraph keeps looping back to THIS node. Without
    # this guard, a permanently-bad query could loop forever, burning
    # tokens and latency indefinitely.
    #
    # JAVA EQUIVALENT — this is a classic RETRY-WITH-MAX-ATTEMPTS guard:
    #     if (currentAttempts >= maxAttempts) {
    #         logger.warn("Max retrieval attempts ({}) reached", maxAttempts);
    #         String fallbackMsg = buildFallbackMessage(maxAttempts);
    #         return Map.of("messages", List.of(new AIMessage(fallbackMsg)));
    #     }
    # Same defensive pattern as a Spring Retry @Recover method that
    # kicks in after @Retryable exhausts its attempts.
    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback_msg = (
            f"I apologize, but I couldn't find relevant research papers after {max_attempts} attempts.\n"
            "This may be because:\n"
            "1. No papers in the database contain relevant information\n"
            "2. The query terms don't match the indexed content\n\n"
            "Please try rephrasing your question with more specific technical terms."
        )

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        # === PYTHON DICT UNPACKING: **updates ===
        # `{**updates, "messages": [...]}` merges the updates dict
        # (which might contain "original_query") with a new key
        # "messages", into ONE combined dict.
        #
        # Java equivalent — manually merging two maps:
        #     Map<String, Object> result = new HashMap<>(updates);
        #     result.put("messages", List.of(new AIMessage(fallbackMsg)));
        #     return result;
        # Python's ** spread operator does this merge in one expression
        # — similar spirit to Java records' "with-style" copying, or
        # Map.of() combined with putAll(), just terser syntax.
        return {**updates, "messages": [AIMessage(content=fallback_msg)]}

    # ============================================================
    # BRANCH B: Normal case — build the tool call request
    # ============================================================
    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Retrieval attempt {new_attempt_count}/{max_attempts}")

    # === THE KEY MOMENT: constructing a TOOL CALL, not calling the tool ===
    #
    # Notice: content="" (empty!) — this AIMessage carries NO text for
    # the user to read. Its entire purpose is the `tool_calls` list,
    # which is a structured request: "please run retrieve_papers with
    # this exact query."
    #
    # JAVA EQUIVALENT — this is literally building a Command object:
    #     ToolCall toolCall = ToolCall.builder()
    #         .id("retrieve_" + newAttemptCount)
    #         .name("retrieve_papers")
    #         .args(Map.of("query", question))
    #         .build();
    #     AIMessage aiMessage = new AIMessage("", List.of(toolCall));
    #
    # WHY THE id FIELD ("retrieve_1", "retrieve_2", ...)?
    # This correlates the REQUEST (this AIMessage's tool_calls[0].id)
    # with the eventual RESPONSE (a ToolMessage with the SAME
    # tool_call_id). Exactly like an HTTP request ID used to match
    # async responses in a message queue — Kafka correlation IDs,
    # or a CompletableFuture keyed by request UUID.
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"retrieve_{new_attempt_count}",
                    "name": "retrieve_papers",  # ← must match the @tool name in tools.py exactly
                    "args": {"query": question},
                }
            ],
        )
    ]

    logger.debug(f"Created tool call for query: {question[:100]}...")

    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "status": "tool_call_created",
                "query": question,
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )

    # === WHAT HAPPENS NEXT (outside this file) ===
    # 1. This AIMessage (with its empty content + tool_calls) gets
    #    appended to state.messages via the add_messages reducer.
    # 2. agentic_rag.py's graph wiring routes from "retrieve" to a
    #    LangGraph ToolNode (built-in, not something we write).
    # 3. ToolNode SEES the tool_calls, looks up "retrieve_papers" in
    #    its registered tools (the one create_retriever_tool built),
    #    and actually invokes it with args={"query": question}.
    # 4. ToolNode wraps the returned List[Document] into a ToolMessage
    #    with tool_call_id="retrieve_1" (matching our request), and
    #    appends THAT to state.messages too.
    # 5. THEN grade_documents_node (next episode after this) reads
    #    that ToolMessage to grade relevance.
    return updates
