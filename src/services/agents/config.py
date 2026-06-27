"""
Configuration for the Agentic RAG graph execution.

=== JAVA COMPARISON ===
This file = your @ConfigurationProperties class + application.yml combined.

In Spring Boot you'd write TWO things:

1. The config class:
    @ConfigurationProperties(prefix = "agent.graph")
    public class GraphConfig {
        private int maxRetrievalAttempts = 2;
        private int guardrailThreshold = 60;
        private String model = "llama3.2:1b";
        // ... getters, setters
    }

2. The YAML:
    agent:
      graph:
        max-retrieval-attempts: 2
        guardrail-threshold: 60
        model: llama3.2:1b

In Python + Pydantic, the class IS the config — defaults live right
in the class definition. No separate YAML file needed. Values can be
overridden at runtime when creating an instance:
    config = GraphConfig(max_retrieval_attempts=3, model="llama3.2:3b")
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

# Settings = our global app config created in earlier weeks
# Think of it as Spring's Environment object — holds all env vars
# like OPENSEARCH_HOST, OLLAMA_HOST, JINA_API_KEY, etc.
#
# get_settings = a factory function that reads .env file and returns Settings
# Java equivalent: @Bean public Settings settings() { return new Settings(); }
from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the Agentic RAG LangGraph workflow.

    Controls how the state machine behaves — retry limits,
    scoring thresholds, model selection, and feature flags.

    === JAVA EQUIVALENT (full mapping) ===

    @ConfigurationProperties(prefix = "agent.graph")
    @Validated
    public class GraphConfig {

        @Value("${agent.graph.max-retrieval-attempts:2}")
        private int maxRetrievalAttempts;

        @Value("${agent.graph.guardrail-threshold:60}")
        private int guardrailThreshold;

        @Value("${agent.graph.model:llama3.2:1b}")
        private String model;

        @Value("${agent.graph.temperature:0.0}")
        private double temperature;

        @Value("${agent.graph.top-k:3}")
        private int topK;

        @Value("${agent.graph.use-hybrid:true}")
        private boolean useHybrid;

        @Value("${agent.graph.enable-tracing:true}")
        private boolean enableTracing;

        private Map<String, Object> metadata = new HashMap<>();

        @Autowired
        private Settings settings;
    }
    """

    # ─── Retry & Threshold Settings ───────────────────────────

    # Java: private int maxRetrievalAttempts = 2;
    #
    # WHY 2? If the first search returns bad results, we rewrite the query
    # and try again. If the SECOND search is also bad, more retries rarely
    # help — the query is probably genuinely unanswerable from our papers.
    # Same reasoning as setting maxRetries=2 in a Spring RestTemplate.
    max_retrieval_attempts: int = 2

    # Java: private int guardrailThreshold = 60;
    #
    # WHY 60? The LLM scores queries 0-100 for "is this about ML papers?"
    # - Below 60: "What is a dog?" (score ~15) → rejected
    # - Above 60: "How does BERT handle typos?" (score ~70) → allowed
    # 60 is tuned empirically. Too low = junk gets through. Too high =
    # valid edge-case queries get rejected. In production, you'd A/B test
    # this threshold using Langfuse traces.
    guardrail_threshold: int = 60

    # ─── LLM Settings ────────────────────────────────────────

    # Java: private String model = "llama3.2:1b";
    #
    # This is the Ollama model name. "1b" = 1 billion parameters.
    # Small enough to run on CPU, fast enough for guardrail decisions.
    # For the final answer generation, you might switch to "3b" or "7b".
    model: str = "llama3.2:1b"

    # Java: private double temperature = 0.0;
    #
    # WHY 0.0? temperature controls randomness in LLM output.
    # 0.0 = deterministic (same input → same output every time)
    # 0.7 = creative (different answers each time)
    #
    # For a RAG system you WANT deterministic — the agent should make
    # the SAME routing decision for the same query every time.
    # Imagine if temperature=0.7 made the guardrail sometimes accept
    # "What is a dog?" and sometimes reject it — that's chaos in prod.
    temperature: float = 0.0

    # ─── Search Settings ─────────────────────────────────────

    # Java: private int topK = 3;
    #
    # How many documents to retrieve from OpenSearch per search.
    # 3 is the sweet spot:
    #   - 1: too few, might miss the best paper
    #   - 3: enough context for a good answer
    #   - 10: too many, LLM gets confused by irrelevant docs
    #         (this is called the "lost in the middle" problem)
    top_k: int = 3

    # Java: private boolean useHybrid = true;
    #
    # hybrid = BM25 (keyword match) + vector (semantic meaning) combined
    # We built this in Week 4. Always better than either alone because:
    #   - BM25 catches exact keyword matches ("BERT", "GPT-4")
    #   - Vector catches semantic meaning ("language model" ≈ "NLP architecture")
    use_hybrid: bool = True

    # ─── Feature Flags ───────────────────────────────────────

    # Java: private boolean enableTracing = true;
    #
    # When true, every node execution gets traced in Langfuse.
    # Turn OFF in unit tests to avoid needing a running Langfuse server.
    # Java equivalent: @ConditionalOnProperty(name = "tracing.enabled")
    enable_tracing: bool = True

    # Java: private Map<String, Object> metadata = new HashMap<>();
    #
    # Runtime metadata passed through to Langfuse traces.
    # Contains: request_id, user_id, session_id, timestamps, etc.
    # Helps debug: "Why was this specific request slow?"
    metadata: Dict[str, Any] = {}

    # Java: @Autowired private Settings settings;
    #
    # Field(default_factory=get_settings) = LAZY initialization
    # The get_settings() function is NOT called when GraphConfig is defined.
    # It's called ONLY when someone accesses config.settings for the first time.
    #
    # Java equivalent: @Lazy @Autowired private Settings settings;
    #
    # get_settings() reads from .env file, same as Spring reading application.yml
    # It returns a Settings object with all env vars: OPENSEARCH_HOST, OLLAMA_HOST, etc.
    settings: Settings = Field(default_factory=get_settings)
