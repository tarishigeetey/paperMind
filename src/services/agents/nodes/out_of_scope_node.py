"""
The Out-of-Scope Node — handles queries the guardrail rejected.

=== JAVA COMPARISON: the big picture ===
This is your "default error handler" — exactly like a Spring
@ExceptionHandler or a fallback @Controller route:

    @ExceptionHandler(OutOfScopeException.class)
    public ResponseEntity<ErrorResponse> handleOutOfScope(OutOfScopeException e) {
        return ResponseEntity.ok(new ErrorResponse(
            "I can only help with CS/AI/ML research questions...",
            e.getQuery()
        ));
    }

When continue_after_guardrail() (Episode 4) routes here instead of to
"retrieve", THIS function is what actually executes — it's the
terminal node for the rejection path.

=== THE KEY DESIGN DECISION IN THIS FILE ===
Remember DIRECT_RESPONSE_PROMPT from prompts.py (Episode 3)? It was
written to have an LLM GENERATE a custom rejection message. But look
at the code below — it does NOT call the LLM at all. It uses a plain
Python f-string template instead.

THIS IS NOT A BUG. It's a real, deliberate production tradeoff:

  LLM-generated rejection (DIRECT_RESPONSE_PROMPT):
    + Sounds more natural/contextual per query
    - Costs an LLM call (latency + token cost) for something that
      doesn't need any "intelligence" — we already KNOW it's rejected
    - introduces a SECOND point of failure (what if THIS LLM call
      also fails? now you need a fallback for your fallback)
    - non-deterministic — testing "did we reject correctly" becomes
      harder when the exact wording changes every time

  Hardcoded template (what's actually wired up):
    + Zero extra latency — this is the FASTEST node in the whole graph
    + Zero extra LLM cost
    + 100% deterministic — easy to unit test exact output
    + Still personalized (echoes back their actual question)

JAVA EQUIVALENT OF THIS TRADEOFF:
Think about validation error responses in a REST API. You could call
an LLM to generate a friendly error message for every 400 Bad Request
("Sorry, your request was malformed because..."), but nobody does
that — you use a STATIC, well-written error message because the
"intelligence" of crafting that response doesn't need to be dynamic.
The guardrail already did the hard thinking (scoring relevance); this
node just needs to communicate a DECISION, not make a new one.

This is the same principle as the "binary vs scored routing" tradeoff
we discussed in prompts.py — the course gives you the LLM-powered
TOOL (DIRECT_RESPONSE_PROMPT) but the actual WIRING uses the cheaper
option. A good interview answer: "I noticed we could call the LLM
again here, but I'd argue against it — the guardrail already
determined this is out of scope, so generating a response is a pure
templating problem, not a reasoning problem. Save the LLM call for
where reasoning is actually needed."
"""

import logging

# Dict, List = Java's Map<K,V> and List<T>
from typing import Dict, List

# AIMessage = the message type representing "the assistant said this"
# Java: an implementation of your Message interface, role="assistant"
from langchain_core.messages import AIMessage

# Runtime[Context] = our DI container wrapper, same as every other node
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_out_of_scope_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Handle out-of-scope queries with a helpful message.

    This node responds to queries that are outside the domain of
    CS/AI/ML research papers with a polite, informative message.

    :param state: Current agent state
    :param runtime: Runtime context (not used in this node)
    :returns: Dictionary with messages containing the out-of-scope response

    NOTICE THE DOCSTRING: "(not used in this node)" next to runtime.
    This is worth flagging — every OTHER node we've seen (guardrail)
    pulls model_name, temperature, langfuse_tracer, etc. out of
    runtime.context. This node takes the SAME runtime parameter (for
    LangGraph's consistent node function signature — every node must
    accept (state, runtime) even if it ignores one) but never touches it.

    JAVA EQUIVALENT: this is like an interface method where one
    implementation just doesn't need all the injected dependencies:
        public interface GraphNode {
            CompletableFuture<Map<String, Object>> invoke(AgentState state, Context ctx);
        }
        // OutOfScopeNode implements GraphNode but ignores `ctx` entirely —
        // it's still required by the interface contract, just unused here.
    """
    logger.info("NODE: out_of_scope")

    # === STEP 1: Get what the user actually asked ===
    # Same utility function from utils.py we built last episode.
    # We need this to ECHO it back in the response — makes the
    # rejection feel personalized instead of a generic canned message.
    question = get_latest_query(state["messages"])

    # === STEP 2: Build the rejection message — PLAIN TEMPLATE, NO LLM ===
    #
    # This is a Python "implicit string concatenation" — when you put
    # multiple string literals next to each other inside parentheses,
    # Python joins them into ONE string at compile time. No + needed.
    #
    # JAVA EQUIVALENT — Java 15+ text block, OR StringBuilder:
    #     String responseText = """
    #         I apologize, but I can only help with questions about academic research papers
    #         in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.
    #
    #         Your question: '%s'
    #
    #         This appears to be outside my domain of expertise. For questions like this, you might want to try:
    #         - General-purpose AI assistants for broad knowledge questions
    #         - Domain-specific resources for topics outside CS/AI/ML
    #         - Technical documentation if asking about specific software/tools
    #
    #         If you have a question about AI/ML research papers, I'd be happy to help!
    #         """.formatted(question);
    #
    # The f-string `f"Your question: '{question}'\n\n"` is Python's
    # string interpolation — directly equivalent to Java's
    # "Your question: '%s'".formatted(question) or a text block's
    # %s placeholder.
    response_text = (
        "I apologize, but I can only help with questions about academic research papers "
        "in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.\n\n"
        f"Your question: '{question}'\n\n"
        "This appears to be outside my domain of expertise. For questions like this, you might want to try:\n"
        "- General-purpose AI assistants for broad knowledge questions\n"
        "- Domain-specific resources for topics outside CS/AI/ML\n"
        "- Technical documentation if asking about specific software/tools\n\n"
        "If you have a question about AI/ML research papers, I'd be happy to help!"
    )

    logger.info("Responding with out-of-scope message")

    # === STEP 3: Return the partial state update ===
    # This is interesting compared to the guardrail node's return value!
    # Guardrail returned: {"guardrail_result": response}  ← a single object
    # THIS node returns:  {"messages": [AIMessage(...)]}   ← a LIST appended to messages
    #
    # WHY messages AND NOT a new field?
    # Remember state.py's add_messages reducer — this is the field
    # that's ALWAYS append-only. By writing the rejection into
    # "messages" as an AIMessage, it becomes part of the conversation
    # history exactly like a normal LLM-generated answer would. The
    # API layer (router) just reads the LAST message in state.messages
    # to build the HTTP response — it doesn't need to know or care
    # whether that last message came from an LLM call or a template.
    #
    # Java equivalent: appending a "canned response" Message object to
    # the same conversation history List<Message> that real AI replies
    # go into — the consumer of that list doesn't need a special case.
    return {"messages": [AIMessage(content=response_text)]}
