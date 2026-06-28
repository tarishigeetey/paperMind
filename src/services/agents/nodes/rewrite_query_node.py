"""
The Rewrite Query Node — closes the retry loop.

=== JAVA COMPARISON: the big picture ===
This is the node that fires when grade_documents_node (Episode 7)
decides routing_decision = "rewrite_query". Think of this as a
QUERY NORMALIZATION / EXPANSION SERVICE that sits in a retry loop:

    @Service
    public class QueryRewriteService {
        public String rewriteForBetterRetrieval(String originalQuery) {
            try {
                QueryRewriteOutput result = callLlmRewriter(originalQuery);
                return result.getRewrittenQuery();
            } catch (Exception e) {
                logger.error("LLM rewrite failed: {}", e.getMessage());
                return originalQuery + " research paper arxiv machine learning";
            }
        }
    }

After this node runs, the graph loops BACK to retrieve_node with the new query — that's the cycle: retrieve → grade → rewrite
→ retrieve → grade → ... up to max_retrieval_attempts times (the
guard we built into retrieve_node).

=== A NOTABLE DESIGN INCONSISTENCY (worth flagging, not hiding) ===
Look closely: this file defines its OWN Pydantic model,
QueryRewriteOutput, right here at the top of the file — instead of
putting it in models.py alongside GuardrailScoring, GradeDocuments,
etc.

WHY DOES THIS MATTER FOR YOU TO NOTICE?
In a code review, you'd flag this: "Why is THIS model defined
locally, when every other structured-output model lives in
models.py?" There's no functional bug here — Python doesn't care
WHERE a class is defined — but it's an inconsistency a senior
engineer would catch:
 QueryRewriteOutput is ONLY ever used
     by this one node, nowhere else in the codebase, unlike
     GuardrailScoring (used by guardrail_node AND continue_after_guardrail)
     or SourceItem (used across multiple nodes + the API response).
     Some teams DO keep "locally-scoped" models next to their only
     consumer, and only "promote" a model to a shared file once a
     SECOND consumer needs it (a real, common refactoring heuristic:
     "don't abstract until you have 2 use cases").
"""

import logging
import time

# Dict, List = Java's Map<K,V> and List<T>
from typing import Dict, List

# HumanMessage = represents "the user said this" — used here in an
# unusual way (see STEP 4 below): we're using it to represent the
# REWRITTEN query, not literally something the human typed.
from langchain_core.messages import HumanMessage

from langgraph.runtime import Runtime

# BaseModel, Field = same Pydantic building blocks from models.py
from pydantic import BaseModel, Field

from ..context import Context
from ..prompts import REWRITE_PROMPT
from ..state import AgentState

logger = logging.getLogger(__name__)


# ============================================================
# Locally-scoped structured output model (see design note above)
# ============================================================
# JAVA EQUIVALENT:
#   public record QueryRewriteOutput(
#       @NotBlank String rewrittenQuery,
#       String reasoning
#   ) {}
# Defined right here because it's ONLY consumed by this one node —
# a reasonable "don't abstract prematurely" choice.
class QueryRewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(description="The improved query optimized for document retrieval")
    reasoning: str = Field(description="Brief explanation of how the query was improved")


async def ainvoke_rewrite_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | List]:
    """Rewrite the original query for better document retrieval using LLM.

    This node uses an LLM to intelligently rewrite the user's query
    to improve the chances of finding relevant documents.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with rewritten_query and updated messages
    """
    logger.info("NODE: rewrite_query")
    start_time = time.time()

    # === STEP 1: Get the ORIGINAL question, with a safety fallback ===
    #
    # `state.get("original_query") or state["messages"][0].content`
    #
    # This is Python's "or" used as a FALLBACK CHAIN — read it as:
    # "use original_query if it's truthy (not None/empty); otherwise,
    # fall back to the very first message's content."
    #
    # JAVA EQUIVALENT:
    #     String originalQuestion = state.getOriginalQuery() != null
    #         ? state.getOriginalQuery()
    #         : state.getMessages().get(0).getContent();
    # or more idiomatically with Optional:
    #     String originalQuestion = Optional.ofNullable(state.getOriginalQuery())
    #         .orElseGet(() -> state.getMessages().get(0).getContent());
    #
    # WHY WOULD original_query EVER BE MISSING? Recall from
    # retrieve_node (Episode 6): it only SETS original_query on the
    # FIRST retrieval attempt (`if state.get("original_query") is None`).
    # This fallback is defensive — in case rewrite_query somehow runs
    # before retrieve_node ever did (shouldn't happen in the wired-up
    # graph, but defensive code doesn't assume the graph is wired
    # perfectly forever).
    original_question = state.get("original_query") or state["messages"][0].content
    current_attempt = state.get("retrieval_attempts", 0)

    logger.debug(f"Rewriting query using LLM: {original_question[:100]}...")

    # === STEP 2: Langfuse span (same pattern as every other node) ===
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="query_rewriting",
                input_data={
                    "original_query": original_question,
                    "attempt": current_attempt,
                },
                metadata={
                    "node": "rewrite_query",
                    "strategy": "llm_based_expansion",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for query rewriting")
        except Exception as e:
            logger.warning(f"Failed to create span for rewrite_query node: {e}")

    # === STEP 3: The actual LLM rewrite call ===
    try:
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            # temperature=0.3 — NOTICE THIS IS DIFFERENT from guardrail
            # and grade_documents, which both use temperature=0.0!
            #
            # WHY 0.3 HERE SPECIFICALLY?
            # Guardrail/grading are CLASSIFICATION tasks — "score this"
            # or "yes/no this" — you want the EXACT same input to always
            # produce the EXACT same output (fully deterministic, 0.0).
            #
            # Query rewriting is closer to a CREATIVE/GENERATIVE task —
            # there's no single "correct" rewritten query, just better
            # or worse ones. A small amount of randomness (0.3, still
            # quite low/focused) can help the LLM avoid getting stuck
            # producing the exact same rewrite twice in a row if this
            # node fires multiple times in one request (2nd rewrite
            # attempt should ideally try a DIFFERENT angle, not repeat
            # the same rewrite verbatim).
            #
            # JAVA EQUIVALENT: like tuning randomness in a recommendation
            # engine — deterministic ranking for "show exact search
            # matches" vs. slight randomization for "suggest something
            # new" so repeated calls don't always return identical results.
            temperature=0.3,
        )
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        prompt = REWRITE_PROMPT.format(question=original_question)

        logger.debug(f"Invoking LLM for query rewriting (model: {runtime.context.model_name})")
        llm_start = time.time()

        result: QueryRewriteOutput = await structured_llm.ainvoke(prompt)

        # === STEP 3a: Validate the LLM's output before trusting it ===
        # WHY EXPLICITLY CHECK THIS, WHEN with_structured_output ALREADY
        # VALIDATES THE SCHEMA?
        # Pydantic validation guarantees rewritten_query is a STRING
        # (the right TYPE) — but it does NOT guarantee it's a
        # NON-EMPTY, MEANINGFUL string. An LLM could technically return
        # rewritten_query="" and still pass Pydantic's schema check
        # (empty string IS a valid str). This extra check catches that
        # "technically valid but practically useless" edge case.
        #
        # JAVA EQUIVALENT — this is the difference between Bean
        # Validation's @NotNull (type-level) and a custom business-rule
        # check like @NotBlank, or an extra `if (result.isBlank())
        # throw new IllegalStateException(...)` after deserialization:
        #     if (result == null || result.getRewrittenQuery() == null) {
        #         throw new IllegalStateException("LLM failed to return valid structured output");
        #     }
        if not result or not result.rewritten_query:
            raise ValueError("LLM failed to return valid structured output for query rewriting")

        # .strip() = Java's .trim() — removes leading/trailing whitespace
        rewritten_query = result.rewritten_query.strip()
        if not rewritten_query:
            raise ValueError("LLM returned empty rewritten query")

        reasoning = result.reasoning

        llm_duration = time.time() - llm_start
        logger.info(f"Query rewritten in {llm_duration:.2f}s: '{original_question[:50]}...' -> '{rewritten_query[:50]}...'")
        logger.debug(f"Rewriting reasoning: {reasoning}")

    # ============================================================
    # THE FALLBACK — what happens if the LLM rewrite totally fails
    # ============================================================
    # WHY THIS SPECIFIC FALLBACK STRATEGY?
    # `f"{original_question} research paper arxiv machine learning"`
    # — this is "dumb keyword stuffing": append generic domain terms
    # to whatever the user originally asked. It's crude, but it's a
    # REASONABLE last resort: even if the LLM can't intelligently
    # rephrase the query, blindly appending "research paper arxiv
    # machine learning" nudges a vector/BM25 search toward our actual
    # corpus instead of drifting toward generic web content.
    #
    # JAVA EQUIVALENT — exactly the same "degrade gracefully, don't
    # crash" philosophy as guardrail's score=50 default (Episode 4)
    # and grade_documents' len(context)>50 heuristic (Episode 7).
    # Every LLM-calling node in this codebase follows the SAME shape:
    # try the smart LLM approach, catch failures, fall back to a
    # crude-but-functional heuristic. That consistency itself is
    # worth recognizing as a deliberate architectural pattern across
    # the whole graph — like a shared @ControllerAdvice exception
    # handler strategy applied uniformly across every endpoint.
    except Exception as e:
        logger.error(f"Failed to rewrite query using LLM: {e}")
        logger.warning("Falling back to simple keyword expansion")
        rewritten_query = f"{original_question} research paper arxiv machine learning"
        reasoning = "Fallback: Simple keyword expansion due to LLM error"

    # === STEP 4: Langfuse span completion ===
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "rewritten_query": rewritten_query,
                "reasoning": reasoning,
                "original_query": original_question,
            },
            metadata={
                "execution_time_ms": execution_time,
                "original_length": len(original_question),
                "rewritten_length": len(rewritten_query),
                # 'llm_duration' in locals() — Python's way to check
                # "does this local variable even exist?" — needed
                # because llm_duration is only ever SET inside the try
                # block; if we hit the except branch, llm_duration was
                # never assigned, so referencing it directly would
                # raise NameError. Java has no equivalent need for this
                # check because Java requires you to declare (and
                # usually initialize) variables before any conditional
                # branch that might use them — the compiler enforces
                # definite assignment. Python has no such compile-time
                # guarantee, so this defensive locals() check exists.
                "llm_duration_seconds": llm_duration if "llm_duration" in locals() else None,
            },
        )

    # === STEP 5: Return the partial state update ===
    #
    # TWO things happen here, and BOTH matter for the retry loop:
    #
    # 1. "rewritten_query": rewritten_query
    #      → stored in AgentState for transparency (so the final API
    #        response CAN show "we searched using: ...")
    #
    # 2. "messages": [HumanMessage(content=rewritten_query)]
    #      → THIS is the one that actually drives the next loop
    #        iteration! Remember get_latest_query() (utils.py, Episode
    #        4) searches messages IN REVERSE for the latest
    #        HumanMessage. By appending a NEW HumanMessage containing
    #        the REWRITTEN query, the next call to retrieve_node will
    #        call get_latest_query() and get THIS rewritten text, not
    #        the user's original wording — even though the user never
    #        actually typed this. We're using the message-history
    #        mechanism as the WAY of "passing forward" the new query
    #        from one node to the next.
    #
    # JAVA EQUIVALENT of this trick — it's like a Saga pattern step
    # writing its output into the SAME shared context field that the
    # NEXT saga step reads its input from, using a generic
    # "currentCommand" slot rather than a dedicated "rewrittenQuery"
    # input parameter. A bit unconventional, but it lets retrieve_node
    # stay completely UNAWARE of whether it's handling the user's
    # original query or a rewritten one — it just always reads "the
    # latest HumanMessage," full stop.
    return {
        "messages": [HumanMessage(content=rewritten_query)],
        "rewritten_query": rewritten_query,
    }
