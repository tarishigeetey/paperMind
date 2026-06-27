"""
Runtime context for the Agentic RAG graph — the Dependency Injection container.

=== JAVA COMPARISON ===
This is EXACTLY Spring's dependency injection, just explicit instead of magic.

In Spring Boot, you'd write:

    @Service
    public class GuardrailNode {
        @Autowired private OllamaClient ollamaClient;
        @Autowired private OpenSearchClient openSearchClient;
        @Autowired private JinaEmbeddingsClient embeddingsClient;
        @Autowired private LangfuseTracer langfuseTracer;

        @Value("${agent.model-name}")     private String modelName;
        @Value("${agent.temperature}")    private double temperature;
        @Value("${agent.guardrail-threshold}") private int guardrailThreshold;
        // Spring's IoC container wires all of this FOR you, invisibly
    }

In LangGraph, there's no IoC container magic — Context is a PLAIN
dataclass and pass explicitly into the graph. Every node function receives it as a
parameter: runtime.context.ollama_client, runtime.context.model_name, etc.

WHY EXPLICIT INSTEAD OF MAGIC @Autowired?
LangGraph nodes are just plain async functions — they're not Spring
@Component-managed beans, so there's no container to inject INTO them
automatically. Context is the explicit "bag of dependencies" you pass
through the Runtime object instead. Slightly more verbose, but you can
SEE exactly what each node depends on just by reading its signature.

=== WHY @dataclass INSTEAD OF Pydantic BaseModel? ===
You've seen BaseModel (models.py, config.py) and TypedDict (state.py).
Now a THIRD pattern: Python's built-in @dataclass.

@dataclass = a plain class with auto-generated __init__, __repr__, __eq__
             NO validation (unlike Pydantic BaseModel)
             NO dict-style access (unlike TypedDict)

WHY NO VALIDATION HERE? Context holds live OBJECTS (client instances,
trace spans) — not raw data. Pydantic validation is for data crossing
a boundary (LLM output, API request body). Once you have a real
OllamaClient instance, there's nothing to "validate" — you're just
holding a reference, like a Java field holding an injected bean.

Java mapping:
    @dataclass        ≈  a plain POJO with @AllArgsConstructor (Lombok)
    Pydantic BaseModel ≈  a record with @Valid (validates incoming data)
    TypedDict          ≈  Map<String, Object> (flexible partial updates)
"""

# dataclass = Python's built-in "plain object with auto-generated boilerplate"
# Java: @AllArgsConstructor @Getter (Lombok) on a POJO
from dataclasses import dataclass

# LangfuseSpan = a single trace span object from the Langfuse SDK
# Java: like a Micrometer Span / OpenTelemetry Span object
from langfuse._client.span import LangfuseSpan

# TYPE_CHECKING = a Python trick to avoid circular imports at runtime
# while still letting your IDE/type-checker see the real type.
# Java has no direct equivalent — Java's import system doesn't have
# this circular-import problem the same way, since Java resolves
# types at compile time across the whole classpath.
from typing import TYPE_CHECKING, Optional

# These are OUR existing service clients — built in earlier weeks (Ollama,
# OpenSearch) and this week (LangfuseTracer for observability).
# Java: these would be @Service or @Component beans
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


# @dataclass decorator auto-generates __init__, __repr__, __eq__ for us
# Java: same as Lombok's @Data or @Value annotation on a class
@dataclass
class Context:
    """Runtime context for agent dependencies — the DI container for our graph.

    This is built ONCE per request (in factory.py) and passed by
    reference into every single node function in the graph. Think of
    it as a "request-scoped Spring bean" that bundles all dependencies
    a node might need, instead of each node grabbing dependencies itself.

    === JAVA EQUIVALENT ===
    @RequestScope  // ← created fresh per HTTP request, like this Context
    public class AgentContext {
        private final OllamaClient ollamaClient;
        private final OpenSearchClient openSearchClient;
        private final JinaEmbeddingsClient embeddingsClient;
        private final LangfuseTracer langfuseTracer;     // nullable
        private LangfuseSpan trace;                       // nullable, mutable
        private boolean langfuseEnabled = false;
        private String modelName = "llama3.2:1b";
        private double temperature = 0.0;
        private int topK = 3;
        private int maxRetrievalAttempts = 2;
        private int guardrailThreshold = 60;

        // Lombok @AllArgsConstructor generates the constructor
        // Lombok @Getter generates all the getters
    }
    """

    # ─── REQUIRED dependencies (no defaults — must be provided) ───
    # In Python, dataclass fields WITHOUT a default value MUST come
    # before fields WITH defaults. Same rule as Java constructor params
    # where you can't have an optional param before a required one.

    # Java: private final OllamaClient ollamaClient;  ← injected, required
    ollama_client: OllamaClient

    # Java: private final OpenSearchClient openSearchClient;
    opensearch_client: OpenSearchClient

    # Java: private final JinaEmbeddingsClient embeddingsClient;
    embeddings_client: JinaEmbeddingsClient

    # Java: private final LangfuseTracer langfuseTracer;  // can be null
    # Optional[LangfuseTracer] = Java's @Nullable LangfuseTracer
    # Tracing might be disabled (enable_tracing=False in GraphConfig),
    # in which case this would be None — same idea as a feature-flagged
    # @Autowired bean that might not be registered.
    langfuse_tracer: Optional[LangfuseTracer]

    # ─── OPTIONAL dependencies (have defaults) ───

    # Java: private LangfuseSpan trace = null;
    # The CURRENT trace span for this specific request. Gets created
    # once at the start of a request and passed down so every node
    # can attach child spans to it (see guardrail_node.py's span logic).
    trace: Optional["LangfuseSpan"] = None

    # Java: private boolean langfuseEnabled = false;
    langfuse_enabled: bool = False

    # Java: private String modelName = "llama3.2:1b";
    model_name: str = "llama3.2:1b"

    # Java: private double temperature = 0.0;
    temperature: float = 0.0

    # Java: private int topK = 3;
    top_k: int = 3

    # Java: private int maxRetrievalAttempts = 2;
    max_retrieval_attempts: int = 2

    # Java: private int guardrailThreshold = 60;
    guardrail_threshold: int = 60
