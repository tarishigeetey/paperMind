"""
The Guardrail Node — the FIRST checkpoint every query passes through.

=== JAVA COMPARISON: the big picture ===
Think of this node as a Spring Security-style FILTER CHAIN entry point:

    @Component
    public class GuardrailFilter implements Filter {
        public FilterResult evaluate(Query query) {
            GuardrailScoring result = callLLMForScore(query);
            if (result.getScore() >= threshold) {
                return FilterResult.CONTINUE;
            }
            return FilterResult.REJECT;
        }
    }

Just like a Spring Security filter decides "let this request through
or 401 it," this node decides "let this query reach retrieval, or
reject it as out-of-scope." Two functions, two responsibilities:

  1. ainvoke_guardrail_step  → DOES the work (calls LLM, gets a score)
  2. continue_after_guardrail → DECIDES the routing (reads the score, branches)

This SEPARATION mirrors a common Java pattern: separate your "do the
work" service method from your "decide what happens next" routing
logic (e.g., a @Service method vs. a @Controller's if/else dispatch).
"""

import logging
import time

# Dict = Java's Map<K, V>
# Literal = Java's enum but inline — restricts return value to exact strings
from typing import Dict, Literal

# Runtime = LangGraph's wrapper that carries our Context (the DI container
# we just built) into every node function call.
# Java: think of Runtime[Context] like Spring's ApplicationContext being
# passed into a method — except LangGraph does it explicitly per-call
# instead of via field injection.
from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


# ============================================================
# FUNCTION 1: continue_after_guardrail (the ROUTING decision)
# ============================================================
# THIS IS A LANGGRAPH "CONDITIONAL EDGE" FUNCTION.
#
# JAVA EQUIVALENT — picture Spring's HandlerInterceptor.preHandle():
#     public boolean preHandle(HttpServletRequest request, ...) {
#         int score = getGuardrailScore(request);
#         if (score >= threshold) {
#             return true;   // continue to controller  ≈ "continue"
#         }
#         response.sendRedirect("/out-of-scope");  ≈ "out_of_scope"
#         return false;
#     }
#
# WHY IS THIS A SEPARATE FUNCTION FROM THE NODE ITSELF?
# In LangGraph, NODES do work and update state. EDGES decide where to
# go next. This separation means agentic_rag.py (the graph wiring file,
# next episode) can plug this exact function in as a conditional edge:
#     graph.add_conditional_edges("guardrail", continue_after_guardrail, {...})
#
# Same separation-of-concerns idea as keeping your @Service business
# logic separate from your @Controller's request-routing logic.
def continue_after_guardrail(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    This function checks the guardrail_result score against a threshold.
    If the score is above threshold, continue; otherwise route to out_of_scope.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise

    JAVA EQUIVALENT:
        public String continueAfterGuardrail(AgentState state, Runtime<Context> runtime) {
            GuardrailScoring guardrailResult = state.getGuardrailResult();
            if (guardrailResult == null) {
                logger.warn("No guardrail result found, defaulting to continue");
                return "continue";
            }
            int score = guardrailResult.getScore();
            int threshold = runtime.getContext().getGuardrailThreshold();
            logger.info("Guardrail score: {}, threshold: {}", score, threshold);
            return score >= threshold ? "continue" : "out_of_scope";
        }
    """
    # state.get("guardrail_result") = Java's Map.get() with implicit null
    # Since AgentState is a TypedDict (= a dict under the hood), we use
    # dict-style .get() instead of attribute access (state.guardrail_result
    # would fail — TypedDict doesn't give you real attributes at runtime,
    # only type hints for your IDE/type-checker).
    guardrail_result = state.get("guardrail_result")

    # Defensive null check — same as Java's:
    #   if (guardrailResult == null) { ...defensive fallback... }
    # WHY DEFAULT TO "continue" INSTEAD OF "out_of_scope"?
    # Fail-open, not fail-closed, for THIS specific check. If the
    # guardrail step somehow never ran (a bug elsewhere), we'd rather
    # let the query through to retrieval (worst case: a slightly
    # irrelevant search) than silently reject every single user query
    # due to an upstream bug. This is a deliberate UX/safety tradeoff —
    # the equivalent of designing a security filter to log-and-continue
    # vs. fail-closed, depending on the blast radius of being wrong.
    if not guardrail_result:
        logger.warning("No guardrail result found, defaulting to continue")
        return "continue"

    # Pull the score out of our Pydantic model (built in models.py)
    # Java: int score = guardrailResult.getScore();
    score = guardrail_result.score

    # runtime.context = our DI container (Context dataclass from context.py)
    # Java: int threshold = runtime.getContext().getGuardrailThreshold();
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail score: {score}, threshold: {threshold}")

    # Python's ternary expression: "A if condition else B"
    # Java: condition ? "continue" : "out_of_scope";
    # Same logic, just the condition sits in the MIDDLE in Python
    # instead of the FRONT like Java's ternary.
    return "continue" if score >= threshold else "out_of_scope"


# ============================================================
# FUNCTION 2: ainvoke_guardrail_step (the WORK — the actual LLM call)
# ============================================================
# "ainvoke" prefix = ASYNC invoke. The "a" prefix is a LangChain/
# LangGraph naming CONVENTION (not a Python language feature) — you'll
# see ainvoke(), abatch(), astream() throughout this codebase, always
# paired with a sync version (invoke(), batch(), stream()) that does
# the same thing but blocks the thread.
#
# JAVA EQUIVALENT — this is exactly a CompletableFuture-returning method:
#     public CompletableFuture<Map<String, GuardrailScoring>> invokeGuardrailStep(
#             AgentState state, Runtime<Context> runtime) {
#         ...
#     }
# "async def" in Python ≈ a method returning CompletableFuture<T> in Java,
# or more precisely, it's like Kotlin's "suspend fun" — non-blocking,
# can be awaited, runs on an event loop instead of blocking a thread.
async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Asynchronously invoke the guardrail validation step using LLM.

    This function evaluates whether the user query is within scope
    (CS/AI/ML research papers) and assigns a score using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with guardrail_result

    NOTICE THE RETURN TYPE: Dict[str, GuardrailScoring], i.e. {"guardrail_result": ...}
    This is a PARTIAL state update — remember from state.py episode,
    LangGraph nodes return ONLY the fields they're changing, and
    LangGraph merges this dict into the full AgentState automatically.
    Java equivalent: like returning a small DTO patch that gets merged
    into a larger ExecutionContext, rather than returning the whole
    rebuilt context object.
    """
    logger.info("NODE: guardrail_validation")

    # time.time() = Java's System.currentTimeMillis() / 1000.0
    # We're manually timing this node for the Langfuse trace below.
    start_time = time.time()

    # === STEP 1: Extract the user's actual question ===
    # state["messages"] = dict-access on our TypedDict (same as state.get(...)
    # above, but here we know the key definitely exists since it's a
    # required field with no default — direct [] access is fine).
    # get_latest_query() = the utility function we just wrote in utils.py
    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluating query: {query[:100]}...")
    # query[:100] = Python slicing, Java: query.substring(0, Math.min(100, query.length()))
    # (Python slicing is forgiving about out-of-range indices — no IndexOutOfBoundsException)

    # === STEP 2: Start a Langfuse observability span (optional) ===
    # This block is OBSERVABILITY code — it doesn't affect business logic,
    # it just records "this node ran, here's what went in, here's what
    # came out, here's how long it took" for the Langfuse dashboard.
    #
    # JAVA EQUIVALENT: this is identical to wrapping a method body in a
    # Micrometer/OpenTelemetry span:
    #     Span span = tracer.spanBuilder("guardrail_validation")
    #         .setAttribute("query", query)
    #         .setAttribute("threshold", threshold)
    #         .startSpan();
    #     try (Scope scope = span.makeCurrent()) {
    #         ...
    #     }
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                input_data={
                    "query": query,
                    "threshold": runtime.context.guardrail_threshold,
                },
                metadata={
                    "node": "guardrail",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for guardrail validation (v2 SDK)")
        except Exception as e:
            # WHY CATCH BROADLY HERE? Tracing should NEVER break the
            # actual feature. If Langfuse is down, we log a warning and
            # move on — same principle as wrapping a metrics/logging call
            # in try-catch in Java so a broken APM agent never takes down
            # your actual business endpoint.
            logger.warning(f"Failed to create span for guardrail validation: {e}")

    # === STEP 3: The actual LLM call (the "business logic") ===
    try:
        # .format(question=query) = fills the {question} placeholder
        # from GUARDRAIL_PROMPT (prompts.py episode)
        # Java: String.format(GUARDRAIL_PROMPT, query);  (positional)
        # or with named params via a templating lib
        guardrail_prompt = GUARDRAIL_PROMPT.format(question=query)

        # Get a LangChain-wrapped Ollama model from our OllamaClient
        # (built in earlier weeks). runtime.context.ollama_client is our
        # injected dependency — same as @Autowired OllamaClient in Spring.
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,  # deterministic — same reasoning as config.py episode
        )

        # === THE KEY LINE: with_structured_output(GuardrailScoring) ===
        # This tells LangChain: "force the LLM's response to conform to
        # this Pydantic schema." Under the hood, LangChain either uses
        # the model's native JSON-mode / function-calling support, or
        # injects formatting instructions into the prompt automatically.
        #
        # JAVA EQUIVALENT: this is like deserializing a REST response
        # straight into a typed DTO:
        #     GuardrailScoring result = objectMapper.readValue(llmRawJson, GuardrailScoring.class);
        # EXCEPT here, the "deserialization contract" is enforced
        # proactively on the LLM's generation itself, not just parsed
        # after the fact — closer to how OpenAPI codegen tools constrain
        # a client to only ever produce schema-valid request bodies.
        structured_llm = llm.with_structured_output(GuardrailScoring)

        # await = Python's "wait for this async call to finish, but
        # don't block the whole thread while waiting" — releases the
        # event loop to handle other requests concurrently.
        # Java: GuardrailScoring response = structuredLlm.ainvoke(prompt).get();
        # (.get() on a CompletableFuture blocks; await does NOT block
        # the underlying thread — much closer to Java's virtual threads
        # or reactive WebFlux's Mono<T>.block() avoided in favor of
        # subscribe()-style non-blocking composition)
        logger.info("Invoking LLM for guardrail validation")
        response = await structured_llm.ainvoke(guardrail_prompt)

        logger.info(f"Guardrail result - Score: {response.score}, Reason: {response.reason}")

        # === STEP 4: Record success in the Langfuse span ===
        if span:
            execution_time = (time.time() - start_time) * 1000  # seconds → ms
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "threshold": runtime.context.guardrail_threshold,
                },
            )

    # === STEP 5: THE FALLBACK — what happens when the LLM call fails ===
    # WHY THIS MATTERS: Ollama might be down, the model might return
    # malformed JSON that fails Pydantic validation, network timeout,
    # etc. WITHOUT this except block, a single guardrail failure would
    # CRASH the entire request with a 500 error.
    #
    # JAVA EQUIVALENT — this is a CIRCUIT BREAKER FALLBACK pattern,
    # exactly like Resilience4j's @CircuitBreaker(fallbackMethod = "..."):
    #     @CircuitBreaker(name = "ollama", fallbackMethod = "guardrailFallback")
    #     public GuardrailScoring callGuardrail(String query) { ... }
    #
    #     public GuardrailScoring guardrailFallback(String query, Exception e) {
    #         logger.error("LLM call failed: {}", e.getMessage());
    #         return new GuardrailScoring(50, "LLM validation failed, using conservative default");
    #     }
    #
    # WHY score=50 AS THE FALLBACK (not 0, not 100)?
    # 50 is right at the edge of our threshold (60) — slightly biased
    # toward REJECTING (since 50 < 60 → out_of_scope). This is a
    # deliberate "fail toward safety" choice: if we can't determine
    # relevance, lean toward NOT spending an expensive retrieval+generation
    # cycle on a query we couldn't validate, rather than letting
    # everything through unchecked.
    except Exception as e:
        logger.error(f"LLM guardrail validation failed: {e}, falling back to default")

        # Fallback to a conservative default if LLM fails
        response = GuardrailScoring(score=50, reason=f"LLM validation failed, using conservative default: {str(e)}")

        # Record the failure in Langfuse too — WARNING level, not ERROR,
        # because we recovered gracefully (the request still completes,
        # just with degraded guardrail accuracy). This distinction
        # matters for alerting: you'd page someone on ERROR, but just
        # track WARNING in a dashboard.
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"score": response.score, "reason": response.reason, "error": str(e)},
                metadata={"execution_time_ms": execution_time, "fallback": True},
                level="WARNING",
            )
            runtime.context.langfuse_tracer.end_span(span)

    # === STEP 6: Return the PARTIAL state update ===
    # Remember: LangGraph merges this dict into AgentState. We're ONLY
    # updating the guardrail_result field — every other field in
    # AgentState stays untouched (messages, retrieval_attempts, etc.)
    #
    # Java: return Map.of("guardrailResult", response);
    # (a tiny patch object, not the whole rebuilt state)
    return {"guardrail_result": response}
