"""
The Generate Answer Node — the happy-path finish line.

=== JAVA COMPARISON: the big picture ===
This is the terminal node for the SUCCESS path — after guardrail
approved the query, retrieve_node found papers, and grade_documents
confirmed they're relevant, THIS node finally produces the answer the
user actually reads.

JAVA EQUIVALENT — think of this as the final @Service call in a
request pipeline, right before the @RestController wraps it into an
HTTP response:

    @Service
    public class AnswerGenerationService {
        public String generateAnswer(String question, String context) {
            try {
                String prompt = GENERATE_ANSWER_PROMPT.formatted(context, question);
                ChatResponse response = llmClient.complete(prompt, modelName, temperature);
                return response.getContent();
            } catch (LlmException e) {
                logger.error("Answer generation failed: {}", e.getMessage());
                return "I apologize, but I encountered an error...";
            }
        }
    }

=== THE BIG STRUCTURAL DIFFERENCE FROM EVERY PRIOR LLM-CALLING NODE ===
Compare the LLM call here vs. guardrail_node (Episode 4) and
grade_documents_node (Episode 7):

    guardrail_node:        llm.with_structured_output(GuardrailScoring)
    grade_documents_node:  llm.with_structured_output(GradeDocuments)
    rewrite_query_node:    llm.with_structured_output(QueryRewriteOutput)
    generate_answer_node:  llm.ainvoke(prompt)   ← NO with_structured_output AT ALL!

WHY THE DIFFERENCE? Every prior node needed a STRUCTURED DECISION —
a score, a yes/no, a rewritten query string in a known field — because
something ELSE in the code (continue_after_guardrail, the routing
ternary, retrieve_node's next call) needed to programmatically READ
that specific field. This node's output is the FINAL ANSWER shown
directly to a human — there's nothing downstream that needs to parse
it into named fields. Free-form natural language IS the correct
output shape here.

JAVA EQUIVALENT OF THIS DISTINCTION: it's exactly the difference
between deserializing a REST response into a typed DTO (when your
code needs to branch on specific fields) versus just returning a raw
String body to render directly in a UI (when a human will read it,
not your code). Forcing structured output here would be like making
a "generate a friendly email" endpoint return JSON instead of letting
it just return the email text — pointless overhead for the actual use case.
"""

import logging
import time

# Dict, List = Java's Map<K,V> and List<T>
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate final answer using retrieved documents.

    This node generates a comprehensive answer to the
    user's question based on the retrieved context using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with messages containing the generated answer

    NOTICE THE RETURN TYPE: Dict[str, List[AIMessage]] — the SAME
    exact shape as out_of_scope_node (Episode 5)! Both nodes are
    TERMINAL nodes (the graph ends right after either one runs) and
    both communicate their result the same way: append one AIMessage
    to the conversation. The API router layer doesn't need a special
    case for "was this a real answer or a rejection" — it just reads
    the last message either way.
    """
    logger.info("NODE: generate_answer")
    start_time = time.time()

    # === STEP 1: Gather everything we need to build the prompt ===
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # state.get("relevant_sources", []) = Java's
    #   state.getOrDefault("relevantSources", Collections.emptyList())
    # We're not actually USING the sources list in the prompt itself
    # (the LLM only sees raw `context` text) — this count is purely
    # for the Langfuse trace metadata, so a dashboard can show "this
    # answer was built from 3 sources" without re-parsing the context.
    sources_count = len(state.get("relevant_sources", []))

    # === STEP 2: Defensive guard — what if context is somehow empty? ===
    # WHY COULD THIS HAPPEN if grade_documents already confirmed
    # relevance? Recall grade_documents_node's fallback heuristic
    # (Episode 7): if the LLM grader itself failed, it falls back to
    # `len(context.strip()) > 50` — meaning a grading_result could be
    # marked "relevant" even when context is borderline thin. This
    # check is a final safety net so the prompt template never gets
    # filled with an empty {context} placeholder, which would produce
    # a confusing or hallucinated answer.
    #
    # JAVA EQUIVALENT:
    #     if (context == null || context.isEmpty()) {
    #         context = "No relevant documents found.";
    #         logger.warn("No context available for answer generation");
    #     }
    if not context:
        context = "No relevant documents found."
        logger.warning("No context available for answer generation")

    logger.debug(f"Generating answer for query: {question[:100]}...")
    logger.debug(f"Using context of length: {len(context)} characters")

    # === STEP 3: Build a logging preview (same pattern as grade_documents_node) ===
    # NOTE: this preview truncates at 1000 chars, NOT 500 like
    # grade_documents_node's preview. Worth noticing as a small,
    # harmless inconsistency — different nodes, slightly different
    # "how much should I log" judgment calls, same as how two
    # different Java services might pick different truncation lengths
    # for their own debug logging without it being a real bug.
    chunks_preview = []
    if context:
        context_preview = context[:1000] + "..." if len(context) > 1000 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # === STEP 4: Langfuse span (same pattern as every prior node) ===
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="answer_generation",
                input_data={
                    "query": question,
                    "context_length": len(context),
                    "sources_count": sources_count,
                    "chunks_used": chunks_preview,
                },
                metadata={
                    "node": "generate_answer",
                    "model": runtime.context.model_name,
                    # NOTICE: this metadata records temperature, while
                    # guardrail/grading nodes' spans didn't bother
                    # logging temperature (since theirs is always a
                    # fixed 0.0, not worth tracking per-call). Here,
                    # runtime.context.temperature is the CONFIGURABLE
                    # value from GraphConfig (Episode 2) — worth
                    # tracking since it could vary across deployments.
                    "temperature": runtime.context.temperature,
                },
            )
            logger.debug("Created Langfuse span for answer generation")
        except Exception as e:
            logger.warning(f"Failed to create span for generate_answer node: {e}")

    # ============================================================
    # STEP 5: THE ACTUAL ANSWER GENERATION — free-form, not structured
    # ============================================================
    try:
        # Fill in {context} and {question} from GENERATE_ANSWER_PROMPT
        # (prompts.py, Episode 3)
        answer_prompt = GENERATE_ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        # runtime.context.temperature — UNLIKE every prior node, this
        # uses the CONFIGURABLE temperature from GraphConfig, not a
        # hardcoded 0.0 or 0.3. Final answer generation is exactly
        # where you'd want to expose a tuning knob to product/users —
        # some teams want strictly factual, deterministic answers
        # (temperature near 0), others want slightly more natural,
        # varied phrasing (temperature 0.3-0.7). This is the ONE node
        # where that choice should be a config decision, not baked
        # into the node's code.
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )

        # === THE KEY DIFFERENCE: plain ainvoke(), no with_structured_output ===
        # We're calling the LLM and just taking its raw text response.
        # Java equivalent: a plain chat completion call where you just
        # read response.getContent() — no JSON deserialization step,
        # because the consumer (a human, via the API/UI) reads
        # natural-language prose directly.
        logger.info("Invoking LLM for answer generation")
        response = await llm.ainvoke(answer_prompt)

        # === Defensive content extraction ===
        # `response.content if hasattr(response, 'content') else str(response)`
        #
        # WHY THE hasattr CHECK? Depending on which LangChain chat
        # model wrapper you're using, ainvoke() might return an
        # AIMessage object (with a .content attribute) OR, in some
        # edge cases / older integrations, a plain string. This
        # defensively handles BOTH shapes without crashing.
        #
        # JAVA EQUIVALENT — handling two possible return shapes from
        # an SDK, similar to:
        #     String answer = (response instanceof ChatResponse cr)
        #         ? cr.getContent()
        #         : response.toString();
        # Java's stricter typing usually prevents this ambiguity at
        # the SDK boundary (the SDK would commit to ONE return type),
        # but Python's more dynamic typing means defensive hasattr()
        # checks like this show up more often at library boundaries.
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Generated answer of length: {len(answer)} characters")

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "answer_length": len(answer),
                    "sources_used": sources_count,
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "context_length": len(context),
                },
            )

    # ============================================================
    # THE FALLBACK — but notice this one is DIFFERENT in spirit
    # ============================================================
    # Compare this fallback to every prior node's fallback:
    #   guardrail:        falls back to a SCORE (50) — still lets the graph continue
    #   grade_documents:   falls back to a HEURISTIC — still lets the graph continue
    #   rewrite_query:     falls back to KEYWORD STUFFING — still lets the graph continue
    #   generate_answer:   falls back to an ERROR MESSAGE shown directly to the user
    #
    # WHY IS THIS ONE DIFFERENT? Every PRIOR node's fallback was
    # designed to let the PIPELINE keep moving — there's always a
    # "next step" downstream that can still do something useful with
    # a degraded-but-valid result. This node IS the last step. There's
    # nothing left to hand off to. So the fallback here has to be
    # HONEST with the user: "something broke, here's what happened,
    # please retry" — rather than silently degrading.
    #
    # JAVA EQUIVALENT — this is the difference between a @Retryable
    # method's @Recover fallback (which returns a still-USABLE
    # degraded result to keep a larger workflow going) versus a
    # top-level @ExceptionHandler in a @RestControllerAdvice (which
    # has no "next step" to hand off to — it must produce the actual
    # final response shown to the client).
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}, falling back to error message")

        answer = f"I apologize, but I encountered an error while generating the answer: {str(e)}\n\nPlease try again or rephrase your question."

        if span:
            execution_time = (time.time() - start_time) * 1000
            # update_span(..., level="ERROR") then end_span(span) —
            # notice this is TWO calls, vs. the success path's single
            # end_span(span, output=...) call. update_span() sets the
            # severity level explicitly to ERROR (so Langfuse's
            # dashboard can filter/alert on it), THEN end_span() closes
            # the span out. Compare to rewrite_query_node's fallback
            # (Episode 8), which did NOT bother updating a separate
            # error level at all — another small inconsistency between
            # nodes, but here it makes sense: a total answer-generation
            # failure is more severe (the user gets literally nothing
            # useful) than a query-rewrite falling back to keyword
            # stuffing (which still produces a usable, if cruder, query).
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="ERROR",
            )
            runtime.context.langfuse_tracer.end_span(span)

    # === STEP 6: Return the final answer as the last message ===
    # Same shape as out_of_scope_node's return (Episode 5) — append
    # ONE AIMessage to the conversation via the add_messages reducer.
    # This message is what the API router (next episode's territory)
    # extracts and sends back as the HTTP response body.
    return {"messages": [AIMessage(content=answer)]}
