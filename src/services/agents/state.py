"""
State definition for the Agentic RAG LangGraph workflow.

=== JAVA COMPARISON ===

In Java/Spring, when you build a multi-step workflow you pass a
"context" or "state" object between steps:

  SPRING BATCH:
    public class StepExecution {
        private ExecutionContext executionContext;  // ← this is a Map
        // each step reads/writes to this context
    }

  SAGA PATTERN:
    public class SagaContext {
        private Map<String, Object> data;
        // each saga step reads from and writes to this
    }

AgentState is EXACTLY this — a shared bag of data that flows between
every node in the LangGraph. Each node reads what it needs, does work,
and writes its results back.

=== WHY TypedDict AND NOT Pydantic BaseModel? ===

You just saw BaseModel in models.py. Why is AgentState a TypedDict instead?

TypedDict = a plain Python dict with type hints (no validation, no methods)
BaseModel = validated object with methods (full validation on every field)

LangGraph REQUIRES TypedDict because nodes return PARTIAL updates:

    # A node can update just ONE field:
    return {"retrieval_attempts": 2}      # ← only this field changes

    # With BaseModel you'd have to return ALL fields every time:
    return AgentState(messages=..., original_query=..., ...)  # ← tedious!

This is the same reason Spring Batch uses ExecutionContext (which is just
a Map<String, Object>) instead of a typed record — steps need to write
partial updates without knowing about every other field.

Java mapping:
    TypedDict ≈ Map<String, Object> with documented keys  (loose, flexible)
    BaseModel ≈ record with @Valid annotations             (strict, validated)

=== THE add_messages REDUCER (critical concept!) ===

The magic line:
    messages: Annotated[list[AnyMessage], add_messages]

That "add_messages" is a REDUCER — it tells LangGraph HOW to merge
this field when a node returns new messages.

WITHOUT reducer:
    Node returns: {"messages": [new_msg]}
    Result: REPLACES the entire message list → old messages LOST! ❌

WITH add_messages reducer:
    Node returns: {"messages": [new_msg]}
    Result: APPENDS new_msg to existing list → history preserved ✅

Java analogy — it's a custom merge strategy:
    context.merge("messages", newMsgs, (oldList, newList) -> {
        oldList.addAll(newList);   // ← this is what add_messages does
        return oldList;
    });

Without this, every node would overwrite the conversation history.
The user's original question would vanish after the first node runs.
"""

# === IMPORTS ===
#
# Annotated = Java's @Qualifier or custom annotation on a type
#   Annotated[list, add_messages] = "this is a list, BUT merge by appending"
#   Java: @MergeStrategy(APPEND) List<Message> messages;
#
# TypedDict = a dict with typed keys — like Map<String, Object> but documented
#   Java: public class ExecutionContext extends HashMap<String, Object> {}
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# AnyMessage = the base type for ALL LangChain messages:
#   - HumanMessage    → what the user said
#   - AIMessage       → what the LLM responded (may include tool_calls)
#   - ToolMessage     → result from a tool execution (e.g., OpenSearch results)
#   - SystemMessage   → system prompt/instructions
#
# Java equivalent:
#   public interface Message {
#       String getContent();
#       String getRole();     // "human", "ai", "tool", "system"
#   }
from langchain_core.messages import AnyMessage

# add_messages = the reducer function that makes message lists append-only
# This is what prevents nodes from accidentally overwriting conversation history
from langgraph.graph.message import add_messages

# Our domain models from models.py (the file we created above)
# These are the Pydantic BaseModel classes used as field types
from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(TypedDict):
    """Shared memory that flows between EVERY node in the LangGraph.

    Think of this as Spring Batch's ExecutionContext —
    every step (node) can read any field and write back updates.

    === LIFECYCLE OF STATE IN ONE REQUEST ===

    User sends: "What are transformers?"

    ┌─ START ────────────────────────────────────────────────────┐
    │ state = {                                                  │
    │   messages: [HumanMessage("What are transformers?")],      │
    │   original_query: "What are transformers?",                │
    │   retrieval_attempts: 0,                                   │
    │   ...everything else is None/empty                         │
    │ }                                                          │
    └────────────────────────────────────────────────────────────┘
         ↓
    ┌─ GUARDRAIL NODE ──────────────────────────────────────────┐
    │ reads:  state["messages"] → extracts query text            │
    │ does:   asks LLM to score 0-100                            │
    │ writes: {                                                  │
    │   guardrail_result: GuardrailScoring(score=85, reason=...) │
    │   routing_decision: RoutingDecision(route="retrieve")      │
    │ }                                                          │
    └────────────────────────────────────────────────────────────┘
         ↓
    ┌─ RETRIEVE NODE ───────────────────────────────────────────┐
    │ reads:  state["messages"] → gets query for OpenSearch      │
    │ does:   calls OpenSearch hybrid search                     │
    │ writes: {                                                  │
    │   messages: [ToolMessage(search_results)],  ← APPENDED!   │
    │   retrieval_attempts: 1,                                   │
    │   sources: {"tool_123": [paper1, paper2, paper3]}          │
    │ }                                                          │
    └────────────────────────────────────────────────────────────┘
         ↓
    ┌─ GRADE DOCUMENTS NODE ────────────────────────────────────┐
    │ reads:  state["sources"] → each paper's content            │
    │ does:   asks LLM "is this relevant?" for each paper        │
    │ writes: {                                                  │
    │   grading_results: [GradingResult(relevant=True), ...],    │
    │   relevant_sources: [SourceItem(title="Attention...")],     │
    │   routing_decision: RoutingDecision(route="generate")      │
    │ }                                                          │
    └────────────────────────────────────────────────────────────┘
         ↓
    ┌─ GENERATE ANSWER NODE ────────────────────────────────────┐
    │ reads:  state["messages"] + state["relevant_sources"]      │
    │ does:   asks LLM to generate final answer with context     │
    │ writes: {                                                  │
    │   messages: [AIMessage("Transformers are...")]  ← APPENDED│
    │ }                                                          │
    └────────────────────────────────────────────────────────────┘
         ↓
    ┌─ END ─────────────────────────────────────────────────────┐
    │ Final state has complete conversation + all metadata       │
    │ API response is built from this final state                │
    └────────────────────────────────────────────────────────────┘


    === JAVA EQUIVALENT ===

    public class AgentState extends HashMap<String, Object> {
        // Typed accessors for compile-time safety:
        public List<Message> getMessages() { ... }
        public String getOriginalQuery() { ... }
        public int getRetrievalAttempts() { ... }
        public GuardrailScoring getGuardrailResult() { ... }
        public RoutingDecision getRoutingDecision() { ... }
        public Map<String, Object> getSources() { ... }
        public List<SourceItem> getRelevantSources() { ... }
        public List<ToolArtefact> getRelevantToolArtefacts() { ... }
        public List<GradingResult> getGradingResults() { ... }
        public Map<String, Object> getMetadata() { ... }
    }
    """

    # ─────────────────────────────────────────────────────────
    # 1. MESSAGE HISTORY (the conversation thread)
    # ─────────────────────────────────────────────────────────
    # This is THE most important field. Holds the full conversation:
    #
    #   [HumanMessage("What are transformers?"),     ← user question
    #    AIMessage(tool_calls=[...]),                 ← agent decided to search
    #    ToolMessage(search_results),                ← OpenSearch returned papers
    #    AIMessage("Transformers are...")]           ← final answer
    #
    # Annotated[..., add_messages] = "when nodes return new messages, APPEND them"
    # Without this, each node would OVERWRITE the entire list.
    #
    # Java: List<Message> messages; with custom merge strategy:
    #   context.merge("messages", newMsgs, (old, nu) -> { old.addAll(nu); return old; });
    messages: Annotated[list[AnyMessage], add_messages]

    # ─────────────────────────────────────────────────────────
    # 2. QUERY TRACKING
    # ─────────────────────────────────────────────────────────
    # We keep BOTH original and rewritten queries so the API response
    # can show: "Original: 'ML stuff' → Rewritten: 'machine learning survey'"
    # This gives users transparency into what the agent did.
    #
    # Java: private String originalQuery;
    original_query: Optional[str]

    # Java: private String rewrittenQuery;
    rewritten_query: Optional[str]

    # ─────────────────────────────────────────────────────────
    # 3. RETRY COUNTER
    # ─────────────────────────────────────────────────────────
    # Tracks how many times we've called OpenSearch.
    # If retrieval_attempts >= config.max_retrieval_attempts (default 2),
    # we stop retrying and generate the best answer we can.
    #
    # Java: private int retrievalAttempts = 0;
    retrieval_attempts: int

    # ─────────────────────────────────────────────────────────
    # 4. GUARDRAIL OUTPUT
    # ─────────────────────────────────────────────────────────
    # Written by the guardrail node: score (0-100) + reason string.
    # The conditional edge reads this to decide: retrieve or reject?
    #
    # Java: private GuardrailScoring guardrailResult;   // nullable
    guardrail_result: Optional[GuardrailScoring]

    # ─────────────────────────────────────────────────────────
    # 5. ROUTING DECISION
    # ─────────────────────────────────────────────────────────
    # Written by guardrail AND grade_documents nodes.
    # Tells the graph which node to visit next.
    # Values: "retrieve", "out_of_scope", "generate_answer", "rewrite_query"
    #
    # Java: private RoutingDecision routingDecision;   // nullable
    routing_decision: Optional[RoutingDecision]

    # ─────────────────────────────────────────────────────────
    # 6. RETRIEVAL RESULTS (raw + filtered)
    # ─────────────────────────────────────────────────────────

    # Raw sources from tool calls: {tool_call_id → [paper1, paper2, ...]}
    # Java: private Map<String, Object> sources;
    sources: Optional[Dict[str, Any]]

    # Filtered sources that PASSED the relevance grading.
    # These are what the user actually sees in the API response.
    # Java: private List<SourceItem> relevantSources = new ArrayList<>();
    relevant_sources: List[SourceItem]

    # Tool execution artifacts with metadata (for debugging/Langfuse tracing)
    # Java: private List<ToolArtefact> relevantToolArtefacts;
    relevant_tool_artefacts: Optional[List[ToolArtefact]]

    # ─────────────────────────────────────────────────────────
    # 7. GRADING RESULTS
    # ─────────────────────────────────────────────────────────
    # One GradingResult per retrieved document.
    # The grade_documents node uses these to decide:
    #   - mostly relevant → route to "generate_answer"
    #   - mostly irrelevant → route to "rewrite_query" (if retries left)
    #
    # Java: private List<GradingResult> gradingResults = new ArrayList<>();
    grading_results: List[GradingResult]

    # ─────────────────────────────────────────────────────────
    # 8. METADATA
    # ─────────────────────────────────────────────────────────
    # Runtime info: request_id, user_id, timestamps, latency, etc.
    # Passed to Langfuse for tracing — helps answer questions like:
    # "Why was this request slow?" or "Which user triggered this query?"
    #
    # Java: private Map<String, Object> metadata = new HashMap<>();
    metadata: Dict[str, Any]
