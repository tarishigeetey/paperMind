"""
Domain models for the Agentic RAG workflow.

=== JAVA COMPARISON ===
This file = your "dto" package in Spring Boot.
Java location: com.papermind.agents.dto

Each class here is equivalent to a Java record/POJO with:
  - Built-in validation (like @Valid, @Min, @Max, @NotNull)
  - Built-in JSON serialization (like Jackson @JsonProperty)
  - Immutable-by-default (like Java 17+ records)

Python's Pydantic BaseModel = Java record + Jackson + Bean Validation
all rolled into one. No separate annotations needed.

Example mapping:
  Python: class Foo(BaseModel):           →  Java: public record Foo(
             name: str = Field(...)                  @NotBlank String name
                                                   ) {}
"""

# === IMPORTS ===
# These are Python's generics — same concept as Java generics
#
# Any       = Java's Object            (can hold anything)
# Dict      = Java's Map<K, V>         (key-value pairs)
# List      = Java's List<T>           (ordered collection)
# Literal   = Java's enum but inline   (restricts to specific string values)
# Optional  = Java's @Nullable         (field can be null/None)
from typing import Any, Dict, List, Literal, Optional

# BaseModel = the base class for ALL Pydantic models
#   Java equivalent: like extending a base record that auto-generates
#   toString(), equals(), hashCode(), JSON serialization, and validation
#
# Field = configures a single field with constraints and metadata
#   Java equivalent: @Min, @Max, @NotNull, @JsonProperty combined
from pydantic import BaseModel, Field


# ============================================================
# 1. GuardrailScoring
# ============================================================
# PURPOSE: When a user sends a query, the FIRST node (guardrail)
#          asks the LLM: "Is this about ML/NLP papers? Score 0-100."
#
#          score >= 60 → proceed to retrieval
#          score < 60  → reject as out-of-scope
#
# JAVA EQUIVALENT:
#   public record GuardrailScoring(
#       @Min(0) @Max(100) int score,    // validated range
#       @NotBlank String reason         // why this score
#   ) {}
#
# WHY THIS EXISTS:
#   Without a guardrail, every query hits OpenSearch + LLM.
#   "What's the weather?" would waste 15 seconds searching ML papers.
#   The guardrail catches this in ~1 second with a lightweight LLM call.
class GuardrailScoring(BaseModel):
    """Scoring result from the guardrail validation node.

    Examples:
        query="What are transformers?"  → score=85, reason="ML architecture topic"
        query="What is a dog?"          → score=15, reason="Not related to ML/NLP"
        query="Tell me about BERT"      → score=90, reason="Specific ML model"
    """

    # Field(ge=0, le=100) = Java's @Min(0) @Max(100)
    # ge = greater-than-or-equal, le = less-than-or-equal
    # If you pass score=150, Pydantic throws ValidationError
    # exactly like Java's ConstraintViolationException
    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")

    # Just a plain required string — no default value means it's mandatory
    # Java: @NotNull String reason
    reason: str = Field(description="Brief reason for the score")


# ============================================================
# 2. GradeDocuments
# ============================================================
# PURPOSE: After OpenSearch returns papers, the grading node asks
#          the LLM for EACH paper: "Is this relevant to the query?"
#          Answer: "yes" or "no"
#
# JAVA EQUIVALENT:
#   public enum BinaryScore { YES, NO }
#   public record GradeDocuments(
#       BinaryScore binaryScore,
#       String reasoning
#   ) {}
#
# WHY Literal["yes", "no"] INSTEAD OF bool?
#   LLMs output text, not true/false.
#   "Output exactly 'yes' or 'no'" is more reliable than "output true or false"
#   because LLMs sometimes output "True", "TRUE", "true" inconsistently.
#   Literal enforces ONLY these two string values — like a Java enum.
class GradeDocuments(BaseModel):
    """Binary relevance score for a single retrieved document."""

    # Literal["yes", "no"] = Java enum with exactly 2 values
    # Pydantic rejects anything else — "maybe" → ValidationError
    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")

    # default="" means this field is optional (defaults to empty string)
    # Java: String reasoning = "";
    reasoning: str = Field(default="", description="Explanation for the decision")


# ============================================================
# 3. SourceItem
# ============================================================
# PURPOSE: Represents ONE arXiv paper in the final API response.
#          The user sees: "Sources: [1] Attention Is All You Need (arxiv.org/...)"
#
# JAVA EQUIVALENT:
#   public record SourceItem(
#       String arxivId,
#       String title,
#       List<String> authors,       // empty list by default
#       String url,
#       double relevanceScore       // 0.0 by default
#   ) {
#       public Map<String, Object> toMap() { ... }
#   }
class SourceItem(BaseModel):
    """One source paper returned to the user in the response."""

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")

    # default_factory=list → creates a NEW empty list for each instance
    # Java equivalent: new ArrayList<>()
    #
    # ⚠️ PYTHON GOTCHA (important for Java devs!):
    # NEVER do: authors: List[str] = []
    # That creates ONE shared list across ALL instances (mutable default).
    # It's like doing: static List<String> authors = new ArrayList<>();
    # Every instance would share the same list! default_factory prevents this.
    authors: List[str] = Field(default_factory=list, description="List of authors")

    url: str = Field(description="Link to paper")

    # default=0.0 → simple immutable default, no factory needed for primitives
    # Java: private double relevanceScore = 0.0;
    relevance_score: float = Field(default=0.0, description="Relevance score from search")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Java equivalent:
            public Map<String, Object> toMap() {
                return Map.of(
                    "arxiv_id", this.arxivId,
                    "title", this.title,
                    "authors", this.authors,
                    "url", this.url,
                    "relevance_score", this.relevanceScore
                );
            }

        NOTE: Pydantic has built-in .model_dump() which is like
        Jackson's ObjectMapper.writeValueAsString(). This custom
        to_dict() gives us control over exact output shape.
        """
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


# ============================================================
# 4. ToolArtefact
# ============================================================
# PURPOSE: When the retrieve node calls OpenSearch (via LangGraph's
#          ToolNode), the raw result gets wrapped in this object.
#          Think of it as a "receipt" of a tool execution — what was
#          called, what ID it got, and what came back.
#
# JAVA EQUIVALENT:
#   public record ToolArtefact(
#       String toolName,            // "retrieve_papers"
#       String toolCallId,          // UUID from LangGraph
#       Object content,             // raw result (any shape)
#       Map<String, Object> metadata // timing, token count, etc.
#   ) {}
#
# WHY "Any" FOR CONTENT?
#   Different tools return different shapes. The retriever returns
#   a list of paper chunks. A future "summarize" tool might return
#   a string. Using Any (= Java Object) keeps this generic.
class ToolArtefact(BaseModel):
    """Artifact returned by a tool call (e.g., OpenSearch retrieval)."""

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")

    # Any = Java's Object — can hold string, dict, list, anything
    content: Any = Field(description="Tool result content")

    # default_factory=dict → new empty dict per instance (same gotcha as list above)
    # Java: new HashMap<>()
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================
# 5. RoutingDecision
# ============================================================
# PURPOSE: After guardrail scores the query, it creates a RoutingDecision
#          that tells the LangGraph WHERE TO GO NEXT.
#          This is the "conditional edge" — the fork in the road.
#
# JAVA EQUIVALENT:
#   public enum Route {
#       RETRIEVE,          // go search OpenSearch
#       OUT_OF_SCOPE,      // reject the query
#       GENERATE_ANSWER,   // skip retrieval, answer directly
#       REWRITE_QUERY      // rephrase and try again
#   }
#   public record RoutingDecision(Route route, String reason) {}
#
# WHY Literal INSTEAD OF Python enum?
#   Pydantic's Literal works directly with JSON — no custom serializer
#   needed. Python enums require extra __json__() methods. For LLM
#   outputs (which are always JSON), Literal is the pragmatic choice.
class RoutingDecision(BaseModel):
    """Where the graph should go next — the 'traffic signal' of our agent."""

    # These 4 values map directly to the 4 possible next-nodes in the graph
    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(description="Next node to route to")
    reason: str = Field(default="", description="Reason for routing decision")


# ============================================================
# 6. GradingResult
# ============================================================
# PURPOSE: Detailed result of grading ONE document.
#          The grade_documents node produces a List[GradingResult].
#          If most docs are irrelevant → routing_decision = "rewrite_query"
#          If most docs are relevant → routing_decision = "generate_answer"
#
# JAVA EQUIVALENT:
#   public record GradingResult(
#       String documentId,
#       boolean isRelevant,
#       double score,
#       String reasoning
#   ) {}
class GradingResult(BaseModel):
    """Result of grading a single retrieved document for relevance."""

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")


# ============================================================
# 7. ReasoningStep
# ============================================================
# PURPOSE: Each node appends a ReasoningStep to explain what it did.
#          The final API response includes ALL steps so the user sees:
#            reasoning_steps: [
#              "Guardrail: score 85/100, ML topic",
#              "Retrieved 3 papers from OpenSearch",
#              "Graded: 2/3 relevant",
#              "Generated answer from relevant papers"
#            ]
#          This is the "transparency" feature — users can debug WHY
#          the agent gave a particular answer.
#
# JAVA EQUIVALENT:
#   public record ReasoningStep(
#       String stepName,                // "guardrail", "retrieve", etc.
#       String description,             // human-readable explanation
#       Map<String, Object> metadata    // latency, token count, etc.
#   ) {}
class ReasoningStep(BaseModel):
    """One step in the agent's reasoning chain — gives transparency to the user."""

    step_name: str = Field(description="Name of the reasoning step")
    description: str = Field(description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")
