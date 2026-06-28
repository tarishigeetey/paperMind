"""
The Grade Documents Node — the quality gate after retrieval.

=== JAVA COMPARISON: the big picture ===
Think of this as a VALIDATION FILTER sitting right after a database
query, before the result reaches the response layer:

    @Service
    public class DocumentGradingService {
        public RoutingDecision gradeAndRoute(String query, String retrievedContext) {
            if (retrievedContext == null || retrievedContext.isBlank()) {
                return RoutingDecision.REWRITE_QUERY;
            }
            GradeDocuments result = callLlmGrader(query, retrievedContext);
            return result.isRelevant() ? RoutingDecision.GENERATE_ANSWER
                                        : RoutingDecision.REWRITE_QUERY;
        }
    }

This node is the SECOND quality gate in the system (the first was the
guardrail in Episode 4). The guardrail asked "should we even search?"
— this one asks "now that we searched, was it actually any good?"

=== IMPORTANT DESIGN NOTE (worth calling out in an interview) ===
Looking at models.py (Episode 2), we defined GradingResult to grade
documents ONE AT A TIME — implying a loop: grade doc 1, grade doc 2,
grade doc 3, then aggregate. But THIS implementation takes a SIMPLER
approach: it concatenates ALL retrieved chunks into one big context
string and asks the LLM ONE question: "is this whole context relevant?"

WHY THE SIMPLER APPROACH WINS HERE:
  Per-document grading (3 separate LLM calls for top_k=3):
    + Precise — you know EXACTLY which of the 3 papers was useless
    - 3x the LLM calls = 3x the latency, 3x the cost
    - More code complexity (looping, aggregating per-doc results)

  Single-blob grading (what's actually implemented):
    + 1 LLM call regardless of top_k — same latency whether you
      retrieved 1 or 10 chunks
    + Simpler routing logic — one yes/no decision
    - Less granular — if 2 of 3 papers are great and 1 is bad, you
      can't selectively drop just the bad one

This is the SAME tradeoff as batch validation vs. per-record
validation in Java: validating a whole CSV file in one pass vs.
validating row-by-row and collecting individual errors. Which one you
pick depends on whether you need GRANULAR diagnostics or just a
binary "is this batch good enough" decision. For our RAG pipeline,
binary is enough — if context.isRelevant() is false, we rewrite the
WHOLE query and re-search anyway, so per-document blame doesn't
change what action we take next.
"""

import logging
import time

# Dict = Java's Map<K,V>
# `str | list` is Python's UNION TYPE syntax (3.10+) — Java's nearest
# equivalent is a sealed interface with two implementations, or
# generics with a wildcard bound.
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradeDocuments, GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results

    JAVA EQUIVALENT SIGNATURE:
        public Map<String, Object> invokeGradeDocumentsStep(
                AgentState state, Runtime<Context> runtime) { ... }

    NOTICE: routing_decision here is set DIRECTLY to a string
    ("generate_answer" / "rewrite_query"), NOT wrapped in the
    RoutingDecision Pydantic model from models.py. Compare this to
    guardrail_node, which routes via a SEPARATE conditional-edge
    function (continue_after_guardrail). This node makes its OWN
    routing decision INLINE and writes the raw string straight into
    state — a slightly different (looser) pattern than Episode 4's.
    Worth noticing: not every node follows the identical structure —
    real codebases (even well-written course code) have some
    inconsistency. Recognizing WHERE patterns diverge, and whether
    that divergence is intentional or just organic drift, is itself
    a useful code-review skill.
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # === STEP 1: Pull the original question + the retrieved context ===
    # get_latest_query()   → the user's question (utils.py, Episode 4)
    # get_latest_context() → the LATEST ToolMessage's content, i.e. the
    #                        concatenated paper chunks retrieve_node's
    #                        tool call produced (utils.py, Episode 4)
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # === STEP 2: Build a logging/observability preview ===
    # This block does NOTHING functional — it's purely for the
    # Langfuse trace, so when you're debugging a bad answer later you
    # can SEE a snippet of what was actually retrieved, without having
    # to dump the entire (possibly huge) context string into your logs.
    #
    # Java equivalent: truncating a large string before logging it,
    # exactly like:
    #     String preview = context.length() > 500
    #         ? context.substring(0, 500) + "..."
    #         : context;
    chunks_preview = []
    if context:
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # === STEP 3: Langfuse span (same pattern every node uses) ===
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    # ============================================================
    # BRANCH A: No context retrieved at all — skip grading entirely
    # ============================================================
    # WHY CAN context BE EMPTY HERE? Recall get_latest_context()'s
    # behavior from Episode 4: it returns "" (not an exception) if no
    # ToolMessage exists yet. This would happen if, e.g., OpenSearch
    # returned ZERO hits for the query — there's nothing to grade, so
    # there's no point asking the LLM "is this relevant?" about an
    # empty string. We short-circuit straight to "rewrite_query."
    #
    # JAVA EQUIVALENT — a classic guard clause / early return:
    #     if (context == null || context.isBlank()) {
    #         logger.warn("No context found, routing to rewrite_query");
    #         return Map.of("routingDecision", "rewrite_query", "gradingResults", List.of());
    #     }
    if not context:
        logger.warning("No context found, routing to rewrite_query")

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        # grading_results: [] — an EMPTY list, not None. Same
        # convention as models.py's default_factory=list pattern:
        # always return a real (possibly empty) collection, never null,
        # so downstream code can safely call len(grading_results)
        # without a null check. Java: return Collections.emptyList();
        return {"routing_decision": "rewrite_query", "grading_results": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    # ============================================================
    # BRANCH B: We have context — ask the LLM to grade it
    # ============================================================
    try:
        # Fill in the {context} and {question} placeholders from
        # GRADE_DOCUMENTS_PROMPT (prompts.py, Episode 3)
        grading_prompt = GRADE_DOCUMENTS_PROMPT.format(
            context=context,
            question=question,
        )

        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        # Same with_structured_output() pattern as guardrail_node
        # (Episode 4) — force the LLM's JSON response to deserialize
        # straight into our GradeDocuments Pydantic model.
        structured_llm = llm.with_structured_output(GradeDocuments)

        logger.info("Invoking LLM for document grading")
        grading_response = await structured_llm.ainvoke(grading_prompt)

        # Python's `==` comparison on strings — works exactly like
        # Java's .equals() (Python has no separate identity-vs-equality
        # gotcha for strings the way Java's == vs .equals() trips
        # people up — Python's == always does value comparison for
        # strings, never reference comparison).
        is_relevant = grading_response.binary_score == "yes"

        # score: 1.0 or 0.0 — a simple binary-to-float mapping, used
        # later for things like averaging across multiple grading
        # calls if this logic gets extended to per-document grading.
        score = 1.0 if is_relevant else 0.0

        logger.info(f"LLM grading: score={grading_response.binary_score}, reasoning={grading_response.reasoning}")

        # Wrap into our GradingResult domain model (models.py, Episode 2)
        grading_result = GradingResult(
            document_id="retrieved_docs",  # placeholder — since we're
            # grading ALL docs as one blob (see the design note at the
            # top of this file), there's no individual document_id to
            # reference. If this were upgraded to per-document grading,
            # each chunk's actual arxiv_id would go here instead.
            is_relevant=is_relevant,
            score=score,
            reasoning=grading_response.reasoning,
        )

    # === THE FALLBACK — same circuit-breaker pattern as guardrail_node ===
    except Exception as e:
        logger.error(f"LLM grading failed: {e}, falling back to heuristic")

        # WHY THIS SPECIFIC FALLBACK HEURISTIC?
        # `len(context.strip()) > 50` — "if we got AT LEAST a
        # reasonable chunk of text back, assume it's probably usable."
        # This is intentionally crude — it's not trying to assess
        # RELEVANCE (that requires understanding, which is exactly
        # what just failed), it's just a sanity check that SOMETHING
        # substantive came back from OpenSearch. Better to attempt an
        # answer from possibly-imperfect context than to force another
        # expensive retry cycle when the LLM grader itself is the thing
        # that's broken (not the retrieval).
        #
        # Java equivalent — a degraded-mode fallback exactly like
        # Resilience4j's fallbackMethod, same spirit as guardrail's
        # score=50 default from Episode 4, just a different heuristic
        # suited to this node's specific failure mode.
        is_relevant = len(context.strip()) > 50
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback heuristic (LLM failed): {'sufficient content' if is_relevant else 'insufficient content'}",
        )

    # === STEP 4: THE ROUTING DECISION — this is the fork in the road ===
    # This single line is the entire reason this node exists. Notice
    # it's a PLAIN Python ternary writing a raw string, not building a
    # RoutingDecision object (compare to guardrail's separate
    # continue_after_guardrail conditional-edge function from Episode 4).
    #
    # Java: String route = isRelevant ? "generate_answer" : "rewrite_query";
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(f"Grading result: {'relevant' if is_relevant else 'not relevant'}, routing to: {route}")

    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
            },
        )

    # === STEP 5: Return the partial state update ===
    # Two fields get updated:
    #   routing_decision  → read by agentic_rag.py's conditional edge
    #                        (next episode) to decide: go to
    #                        generate_answer_node OR rewrite_query_node
    #   grading_results    → a LIST containing this one grading_result
    #                        (wrapped in a list because AgentState.grading_results
    #                        is typed as List[GradingResult] — even
    #                        though we only produced ONE result here,
    #                        the field's shape supports future
    #                        per-document grading without a schema change)
    return {
        "routing_decision": route,
        "grading_results": [grading_result],
    }
