"""
AgenticRAGService — wires all 7 nodes into one runnable StateGraph.

=== JAVA COMPARISON: the big picture ===
Every episode so far built ONE PIECE (a node, a model, a prompt). This
file is where they all get ASSEMBLED into an actual executable
workflow — think of it as the @Configuration class that wires
together a Spring State Machine, or more precisely, like building a
Spring Batch Job out of individual Steps:

    @Configuration
    public class AgenticRagJobConfig {
        @Bean
        public Job agenticRagJob(JobBuilderFactory jobs, StepBuilderFactory steps) {
            return jobs.get("agenticRagJob")
                .start(guardrailStep())
                .next(decideRoute())              // conditional branching
                    .on("CONTINUE").to(retrieveStep())
                    .on("OUT_OF_SCOPE").to(outOfScopeStep()).end()
                .from(retrieveStep()).next(toolRetrieveStep())
                .next(gradeDocumentsStep())
                .next(decideAfterGrading())        // conditional branching
                    .on("RELEVANT").to(generateAnswerStep())
                    .on("NOT_RELEVANT").to(rewriteQueryStep())
                .from(rewriteQueryStep()).next(retrieveStep())  // LOOP back
                .from(generateAnswerStep()).next(verifyCitationsStep()).end()
                .build();
        }
    }

LangGraph's StateGraph IS this same concept — a fluent builder
(.add_node(), .add_edge(), .add_conditional_edges()) that assembles a
directed graph of steps, then COMPILES it into a runnable object
(graph.ainvoke()) — exactly like Spring Batch's JobBuilder.build()
produces a runnable Job.
"""

import logging
import time

# Dict, List, Optional = Java's Map<K,V>, List<T>, @Nullable
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

# CallbackHandler = Langfuse's auto-tracing hook for LangChain calls.
# Java equivalent: an AOP @Around advice that wraps every LLM call
# automatically to record timing/IO without each node having to
# manually instrument itself (separate from our manual span creation
# inside each node — this is a SECOND, complementary layer of tracing).
from langfuse.langchain import CallbackHandler

# START, END = special sentinel nodes marking the graph's entry/exit
# StateGraph = the builder class for constructing a LangGraph workflow
# Java: START/END ≈ Spring Batch's implicit job start/FlowExecutionStatus.COMPLETED
#       StateGraph ≈ JobBuilder
from langgraph.graph import END, START, StateGraph

# ToolNode = LangGraph's BUILT-IN node that knows how to execute ANY
# tool_calls found in the latest AIMessage — this is the component we
# referenced back in Episode 6 ("a separate LangGraph component...")
# tools_condition = a BUILT-IN conditional-edge function that checks
# "does the latest message have tool_calls? if yes, route to tools;
# if no, route to END" — we don't write this ourselves, LangGraph ships it.
#
# Java equivalent: think of ToolNode like Spring's
# @Async @EventListener dispatcher that inspects an event object and
# calls the right handler — except here it's "inspect tool_calls,
# call the matching @tool-decorated function."
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    ainvoke_verify_citations_step,
    continue_after_guardrail,
)
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions

    === JAVA COMPARISON ===
    This class is your @Service facade — the SAME role as a Spring
    @Service that wraps a complex workflow behind a simple public API
    (here, just one real method: .ask(query)). Internally it builds
    and owns a compiled LangGraph, similar to how a Spring @Service
    might build and own a Spring Batch Job or a Camel route at
    construction time, then expose a single execute() method.
    """

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        ollama_client: OllamaClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param opensearch_client: Client for document search
        :param ollama_client: Client for LLM generation
        :param embeddings_client: Client for embeddings
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution

        JAVA EQUIVALENT (constructor injection — exactly Spring's
        recommended DI style, just without @Autowired magic):
            public AgenticRAGService(
                    OpenSearchClient opensearchClient,
                    OllamaClient ollamaClient,
                    JinaEmbeddingsClient embeddingsClient,
                    @Nullable LangfuseTracer langfuseTracer,
                    @Nullable GraphConfig graphConfig) {
                this.opensearch = opensearchClient;
                this.ollama = ollamaClient;
                this.embeddings = embeddingsClient;
                this.langfuseTracer = langfuseTracer;
                this.graphConfig = graphConfig != null ? graphConfig : new GraphConfig();
                this.graph = buildGraph();   // build ONCE at construction
            }
        """
        self.opensearch = opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer

        # `graph_config or GraphConfig()` — same fallback-to-default
        # idiom we saw in rewrite_query_node (Episode 8).
        # Java: this.graphConfig = graphConfig != null ? graphConfig : new GraphConfig();
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        # === BUILD THE GRAPH ONCE, AT CONSTRUCTION TIME ===
        # WHY BUILD IT HERE, NOT PER-REQUEST? Building the graph (all
        # the add_node/add_edge calls) is pure WIRING — it doesn't
        # depend on any per-request data (the user's actual query).
        # Building it once and reusing the compiled graph object across
        # every .ask() call is both faster (no rebuild overhead per
        # request) and the correct architectural pattern — same reason
        # a Spring Batch Job bean is built ONCE at application startup,
        # not reconstructed for every job execution; only the
        # PER-EXECUTION state (JobParameters, our AgentState) varies.
        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self):
        """Build and compile the LangGraph workflow.

        Uses context_schema for type-safe dependency injection.
        Nodes are lightweight functions that receive Runtime[Context].

        :returns: Compiled graph ready for invocation
        """
        logger.info("Building LangGraph workflow with context_schema")

        # === Create the StateGraph builder ===
        # StateGraph(AgentState, context_schema=Context) tells
        # LangGraph: "every node will read/write THIS state shape
        # (AgentState from state.py, Episode 2), and every node can
        # expect a Runtime[Context] (context.py, Episode 4) injected."
        #
        # Java equivalent — declaring the generic types for a builder:
        #     JobBuilder<AgentState, Context> workflow = new JobBuilder<>(AgentState.class, Context.class);
        workflow = StateGraph(AgentState, context_schema=Context)

        # === Create the retriever tool (Episode 6's factory function) ===
        # This is called ONCE here, capturing opensearch/embeddings/
        # top_k/use_hybrid via closure — exactly as designed in tools.py.
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )
        tools = [retriever_tool]

        # ============================================================
        # ADD NODES — registering each function under a string name
        # ============================================================
        # workflow.add_node("name", function) = Java's
        #     job.addStep("name", new StepBuilder().tasklet(function).build());
        #
        # Notice we're passing the FUNCTION ITSELF (e.g.
        # ainvoke_guardrail_step), not calling it. LangGraph will call
        # it later, once per request, with (state, runtime) arguments.
        # Java equivalent: passing a method reference, like
        #     job.addStep("guardrail", GuardrailNode::invoke);
        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)

        # ToolNode(tools) — THIS is LangGraph's BUILT-IN executor we
        # referenced in Episode 6's "what happens next" comment. We
        # never wrote a function for this node ourselves — LangGraph
        # provides it, and it automatically knows how to: read the
        # latest AIMessage's tool_calls, find the matching tool by
        # NAME (our retriever_tool is named "retrieve_papers" via the
        # @tool decorator), invoke it with the args, and wrap the
        # result into a ToolMessage with the matching tool_call_id.
        workflow.add_node("tool_retrieve", ToolNode(tools))

        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        # NEW NODE: citation verification, runs after generate_answer,
        # before END. Detects and corrects hallucinated citations using
        # the actual retrieved context as ground truth (see
        # verify_citations_node.py for the full implementation).
        workflow.add_node("verify_citations", ainvoke_verify_citations_step)

        # ============================================================
        # ADD EDGES — the actual flow control
        # ============================================================
        logger.info("Configuring graph edges and routing logic")

        # --- START → guardrail (every request begins here) ---
        # Java: job.start(guardrailStep());
        workflow.add_edge(START, "guardrail")

        # --- Guardrail → conditional branch (Episode 4's edge function) ---
        # add_conditional_edges(from_node, decision_function, mapping)
        # — decision_function returns a STRING, and `mapping` translates
        # that string into the ACTUAL next node name.
        #
        # continue_after_guardrail (built in Episode 4) returns either
        # "continue" or "out_of_scope" — this dict maps those exact
        # strings to real node names ("retrieve" / "out_of_scope").
        #
        # JAVA EQUIVALENT — Spring Batch's conditional flow:
        #     job.from(guardrailStep())
        #         .on("CONTINUE").to(retrieveStep())
        #         .on("OUT_OF_SCOPE").to(outOfScopeStep()).end();
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )

        # --- Out of scope → END (terminal node, Episode 5) ---
        # Java: .end();  — this branch of the flow terminates here.
        # NOTE: out_of_scope does NOT go through verify_citations — a
        # rejection message never contains citations to verify, so
        # routing it straight to END avoids a pointless no-op check.
        workflow.add_edge("out_of_scope", END)

        # --- Retrieve → conditional branch using LangGraph's BUILT-IN ---
        # tools_condition is NOT something we wrote — LangGraph ships
        # it. It inspects: "does the latest AIMessage have tool_calls?"
        #   - YES → returns "tools" → routes to "tool_retrieve" (ToolNode)
        #   - NO  → returns END     → routes straight to END
        #
        # WHY WOULD retrieve_node EVER produce NO tool_calls, going
        # straight to END? Recall Episode 6's "max attempts reached"
        # branch — in that case, retrieve_node returns a plain
        # AIMessage with NO tool_calls (just the fallback text). That's
        # exactly the case tools_condition catches here, routing
        # straight to END instead of trying to execute a (non-existent)
        # tool call.
        #
        # JAVA EQUIVALENT — this is the SAME pattern as guardrail's
        # conditional edge, just using a LIBRARY-PROVIDED decision
        # function instead of one we wrote ourselves:
        #     job.from(retrieveStep())
        #         .on("HAS_TOOL_CALLS").to(toolRetrieveStep())
        #         .on("NO_TOOL_CALLS").end();
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )

        # --- After tool execution → always grade what came back ---
        # Java: job.from(toolRetrieveStep()).next(gradeDocumentsStep());
        # This is a SIMPLE edge (add_edge), not conditional — there's
        # only ONE possible next step after a tool call completes.
        workflow.add_edge("tool_retrieve", "grade_documents")

        # --- After grading → conditional branch (Episode 7's INLINE routing) ---
        # NOTICE: unlike guardrail's edge (which used a NAMED function,
        # continue_after_guardrail), this one uses an INLINE LAMBDA:
        #     lambda state: state.get("routing_decision", "generate_answer")
        #
        # This directly reads the routing_decision STRING that
        # grade_documents_node (Episode 7) wrote straight into state
        # — remember, that node set the route INLINE rather than via a
        # separate conditional-edge function, unlike guardrail's split
        # design. This lambda is just reading that pre-computed value
        # back out, with "generate_answer" as a safety-net default if
        # routing_decision somehow wasn't set at all.
        #
        # JAVA EQUIVALENT — an inline lambda for trivial dispatch logic,
        # vs. a named method reference for more complex/reusable logic:
        #     job.from(gradeDocumentsStep())
        #         .on(state -> state.getOrDefault("routingDecision", "generate_answer"))
        #         ... (Java doesn't have first-class lambda-as-edge-condition
        #         in Spring Batch quite this fluidly, but conceptually
        #         this is an inline Function<State,String> vs. a
        #         dedicated @Component class doing the same job)
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        # --- After rewriting → loop BACK to retrieve ---
        # THIS IS THE EDGE THAT CLOSES THE RETRY LOOP we visualized in
        # Episode 8! rewrite_query (Episode 8) → retrieve (Episode 6)
        # → tool_retrieve → grade_documents (Episode 7) → either
        # generate_answer OR back to rewrite_query again — bounded by
        # the max_retrieval_attempts guard built into retrieve_node.
        #
        # Java: job.from(rewriteQueryStep()).next(retrieveStep());
        # — a CYCLE in the flow graph, which Spring Batch supports via
        # exactly this kind of explicit "loop back" transition.
        workflow.add_edge("rewrite_query", "retrieve")

        # --- After answer generation → verify citations, THEN done ---
        # CHANGED: this used to go straight to END. Now it routes
        # through verify_citations first — the hallucination-correction
        # node — before the request actually completes. Every generated
        # answer gets checked; out_of_scope rejections (above) skip this
        # entirely since they never contain citations.
        workflow.add_edge("generate_answer", "verify_citations")
        workflow.add_edge("verify_citations", END)

        # === Compile — converts the builder into a runnable object ===
        # Java: Job agenticRagJob = jobBuilder.build();
        # This is the moment LangGraph validates the graph structure
        # (e.g., checks every conditional edge's possible outputs map
        # to a REAL node) and produces a compiled, invokable graph —
        # exactly like Spring Batch's JobBuilder.build() validates and
        # produces a runnable Job bean.
        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("✓ Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty

        JAVA EQUIVALENT — the public entrypoint of our @Service facade:
            public AgenticRagResult ask(String query, String userId, String modelOverride) {
                if (query == null || query.isBlank()) {
                    throw new IllegalArgumentException("Query cannot be empty");
                }
                // ... build trace, run workflow, extract results
            }
        """
        model_to_use = model or self.graph_config.model

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # === Input validation — fail fast, exactly like Java would ===
        # Java: if (query == null || query.strip().isEmpty()) { throw new IllegalArgumentException(...); }
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        # === Start a Langfuse TRACE (the parent of all the SPANS each
        # node creates) ===
        # Recall: every node we built creates a SPAN as a CHILD of this
        # trace (runtime.context.trace). This is the request-level
        # "root" span — one trace per .ask() call, containing every
        # node's span nested inside it. In a Langfuse dashboard, you'd
        # see ONE trace entry per user request, expandable to show all
        # 6 nodes' individual spans with their timings.
        #
        # Java equivalent — exactly a root OpenTelemetry Span that
        # child spans (from each "step") attach to:
        #     Span trace = tracer.spanBuilder("agentic_rag_request").startSpan();
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.client:
            logger.info("Creating Langfuse trace (v3 SDK)")
            metadata = {
                "env": self.graph_config.settings.environment,
                "service": "agentic_rag",
                "top_k": self.graph_config.top_k,
                "use_hybrid": self.graph_config.use_hybrid,
                "model": model_to_use,
            }
            # start_as_current_span — designed to be used as a context
            # manager (Python's `with` statement, see _execute_with_trace
            # below). Java: try (Scope scope = trace.makeCurrent()) { ... }
            trace = self.langfuse_tracer.client.start_as_current_span(
                name="agentic_rag_request",
            )

        # === Nested async helper — handles the "with or without
        # tracing" branching cleanly ===
        # WHY A NESTED FUNCTION HERE INSTEAD OF INLINE try/except?
        # Python's `with trace as trace_obj:` (a CONTEXT MANAGER) needs
        # its body INDENTED inside the `with` block. If tracing is
        # DISABLED (trace is None), there's no `with` block to enter at
        # all — we just call _run_workflow directly. Wrapping BOTH
        # branches in a small async function keeps the actual call to
        # self._run_workflow() written ONCE conceptually, even though
        # it's technically called from two different code paths.
        #
        # JAVA EQUIVALENT — exactly the same shape as a try-with-resources
        # block that's OPTIONAL depending on a feature flag:
        #     CompletableFuture<Result> executeWithTrace() {
        #         if (trace != null) {
        #             try (Scope scope = trace.makeCurrent()) {
        #                 return runWorkflow(query, modelToUse, userId, trace);
        #             }
        #         }
        #         return runWorkflow(query, modelToUse, userId, null);
        #     }
        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if trace is not None:
                # `with trace as trace_obj:` — Python's context manager
                # protocol. Java's closest equivalent is
                # try-with-resources: `try (var traceObj = trace) { ... }`
                # — entering the `with` block calls trace.__enter__()
                # (sets this span as "current"), and exiting calls
                # trace.__exit__() (cleans up), similar to how
                # AutoCloseable.close() fires automatically.
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    logger.debug(f"Trace created: {trace_obj}")
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj)
            else:
                return await self._run_workflow(query, model_to_use, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace) -> dict:
        """Execute the workflow with the given trace context.

        JAVA EQUIVALENT — this is the actual "run the Spring Batch Job"
        call, equivalent to:
            JobExecution execution = jobLauncher.run(agenticRagJob, jobParameters);
        """
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            # === STEP 1: Build the INITIAL AgentState ===
            # Remember AgentState is a TypedDict (Episode 2) — this is
            # literally constructing the starting dict that flows
            # through every node. EVERY field from state.py needs an
            # initial value here (even if it's None/empty) — TypedDict
            # doesn't auto-fill missing keys the way a Pydantic
            # BaseModel would apply Field defaults.
            #
            # Java equivalent — building the JobParameters / initial
            # ExecutionContext before launching a Spring Batch job:
            #     Map<String, Object> initialState = new HashMap<>();
            #     initialState.put("messages", List.of(new HumanMessage(query)));
            #     initialState.put("retrievalAttempts", 0);
            #     ... etc, one entry per AgentState field
            state_input = {
                "messages": [HumanMessage(content=query)],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            }

            # === STEP 2: Build the Runtime Context (DI container, Episode 4) ===
            # This is constructed ONCE PER REQUEST (unlike self.graph,
            # built once at construction time) because it carries
            # per-request data: THIS specific trace, THIS user's model
            # override, etc. Java equivalent: a @RequestScope bean,
            # freshly created for every incoming HTTP request.
            runtime_context = Context(
                ollama_client=self.ollama,
                opensearch_client=self.opensearch,
                embeddings_client=self.embeddings,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.client is not None,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                top_k=self.graph_config.top_k,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            # === STEP 3: LangGraph's own thread/session ID ===
            # `thread_id` lets LangGraph (and Langfuse) correlate
            # multiple graph invocations belonging to the SAME
            # conversation/session — relevant if you ever add
            # multi-turn memory (LangGraph's checkpointing feature,
            # not used here, but this is the hook for it).
            # Java equivalent: a sessionId or conversationId passed
            # through a distributed trace context (e.g., a B3/W3C
            # trace header in a microservices call chain).
            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            # === STEP 4: Attach Langfuse's AUTO-tracing callback ===
            # CallbackHandler() — this is a SECOND, complementary layer
            # of tracing on top of the manual spans each node creates.
            # It automatically intercepts EVERY raw LLM call LangChain
            # makes (the actual llm.ainvoke() calls inside guardrail,
            # grading, rewrite, generate nodes) and records them,
            # WITHOUT each node needing to manually wrap that one call.
            #
            # Java equivalent — this is like adding a
            # MeterRegistry/Micrometer @Timed annotation automatically
            # via Spring AOP, layered ON TOP of manual custom spans you
            # already created — belt-and-suspenders observability.
            if self.langfuse_tracer and trace:
                try:
                    callback_handler = CallbackHandler()
                    config["callbacks"] = [callback_handler]
                    logger.info("✓ CallbackHandler added (will auto-link to current trace)")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            # ============================================================
            # STEP 5: THE ACTUAL EXECUTION — run the compiled graph
            # ============================================================
            # self.graph (built ONCE in __init__, via _build_graph())
            # gets invoked HERE, per-request, with:
            #   state_input     → the starting AgentState
            #   config          → thread_id + callback handlers
            #   context         → our Context DI container (Episode 4)
            #
            # LangGraph takes it from here: starts at START, runs
            # guardrail, follows whichever edges the conditional
            # functions select, loops through retrieve/grade/rewrite as
            # needed, runs verify_citations after generate_answer, and
            # returns the FINAL AgentState once it reaches END.
            #
            # Java equivalent:
            #     JobExecution execution = jobLauncher.run(agenticRagJob, jobParameters);
            #     ExecutionContext finalState = execution.getExecutionContext();
            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # === STEP 6: Extract the bits the API response actually needs ===
            # `result` is the FULL final AgentState — a big dict with
            # every field. These three helper methods (defined below)
            # pull out just what the caller (the FastAPI router, coming
            # next episode) needs to build a clean JSON response.
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            # === STEP 7: Close out the Langfuse trace with final output ===
            if trace:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                trace.end()
                self.langfuse_tracer.flush()

            logger.info("=" * 80)
            logger.info("Agentic RAG Request Completed Successfully")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Sources found: {len(sources)}")
            logger.info(f"Retrieval attempts: {retrieval_attempts}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info("=" * 80)

            # === STEP 8: Build the final response dict ===
            # Java: return new AgentRagResult(query, answer, sources, ...);
            # Python here just returns a plain dict — the FastAPI
            # router will validate/serialize this into a proper
            # Pydantic response model for the actual HTTP response,
            # similar to a Java @Service returning a domain object
            # that a @RestController then wraps in ResponseEntity.
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "reasoning_steps": reasoning_steps,
                "retrieval_attempts": retrieval_attempts,
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                # Nested ternary-like Python expression:
                # `result.get("guardrail_result").score if result.get("guardrail_result") else None`
                # Java: result.getGuardrailResult() != null ? result.getGuardrailResult().getScore() : null;
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                trace.end()
                self.langfuse_tracer.flush()

            raise

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result.

        JAVA EQUIVALENT:
            private String extractAnswer(Map<String, Object> result) {
                List<Message> messages = (List<Message>) result.getOrDefault("messages", List.of());
                if (messages.isEmpty()) return "No answer generated.";
                Message finalMessage = messages.get(messages.size() - 1);
                return finalMessage.getContent();
            }

        WHY messages[-1] (the LAST message) IS ALWAYS THE ANSWER:
        Recall both terminal paths — out_of_scope_node (Episode 5) and
        generate_answer_node → verify_citations_node (Episode 9 + the
        hallucination fix) — ALL return their result as
        `{"messages": [AIMessage(content=...)]}`. Whichever path the
        graph took, the LAST message in the final state IS the
        user-facing answer, by construction. This function doesn't
        need an if/else for "was it a rejection, a generated answer,
        or a corrected answer" — the append-only messages design
        (add_messages reducer, Episode 2) makes all cases uniform.
        """
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        # Python's negative indexing: messages[-1] = "last element"
        # Java: messages.get(messages.size() - 1);
        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result.

        JAVA EQUIVALENT:
            private List<Map<String, Object>> extractSources(Map<String, Object> result) {
                List<Object> relevantSources = (List<Object>) result.getOrDefault("relevantSources", List.of());
                List<Map<String, Object>> sources = new ArrayList<>();
                for (Object source : relevantSources) {
                    if (source instanceof SourceItem si) {
                        sources.add(si.toMap());
                    } else if (source instanceof Map) {
                        sources.add((Map<String, Object>) source);
                    }
                }
                return sources;
            }

        WHY THE DEFENSIVE hasattr/isinstance CHECK? relevant_sources
        is typed as List[SourceItem] in state.py, so in the "should
        always work" case, every item has .to_dict() (the method we
        wrote on SourceItem back in models.py, Episode 2). The isinstance(source, dict)
        fallback is defensive — handling a case where something
        upstream put a plain dict in there instead of a real
        SourceItem instance (type hints in Python are NOT enforced
        at runtime, unlike Java's compiler-checked types, so this kind
        of defensive runtime check is a recurring theme you'll see
        throughout Python codebases at API/data boundaries).
        """
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result.

        JAVA EQUIVALENT — building a human-readable audit trail string
        list from the final state, exactly like assembling a list of
        "what happened" log lines for a UI to display:
            private List<String> extractReasoningSteps(Map<String, Object> result) {
                List<String> steps = new ArrayList<>();
                ...
                return steps;
            }

        NOTE: This builds plain STRINGS, not ReasoningStep objects (the
        Pydantic model we defined in models.py, Episode 2!). Another
        small inconsistency worth flagging — ReasoningStep exists in
        the domain model but isn't actually used here; this method
        does its own ad-hoc string formatting instead. Not a bug, just
        evidence that ReasoningStep may be a "designed but not yet
        wired up" piece — exactly the kind of thing you'd clean up in
        a real refactor pass.
        """
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            # Python generator expression inside sum() — a compact way
            # to count matching items.
            # Java: long relevantCount = gradingResults.stream().filter(GradingResult::isRelevant).count();
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        steps.append("Generated answer from context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Get the LangGraph workflow visualization as PNG.

        :returns: PNG image bytes
        :raises ImportError: If required dependencies (pygraphviz/graphviz) are not installed

        JAVA COMPARISON: there's genuinely no common Java equivalent
        for this — generating a live diagram FROM a running workflow
        definition isn't typical in Spring Batch tooling (you'd more
        likely hand-draw a sequence diagram separately). This is one
        of LangGraph's nicer built-in features: self.graph.get_graph()
        introspects the compiled StateGraph's actual node/edge
        structure and renders it, so the diagram can NEVER drift out
        of sync with the real wiring — unlike a hand-maintained
        architecture diagram that can silently go stale.
        """
        try:
            logger.info("Generating graph visualization as PNG")
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"✓ Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            logger.error(f"Failed to generate visualization - missing dependencies: {e}")
            logger.error("Install with: pip install pygraphviz or apt-get install graphviz")
            raise ImportError(
                "Graph visualization requires pygraphviz. Install with: pip install pygraphviz (requires graphviz system package)"
            ) from e
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}")
            raise

    def get_graph_mermaid(self) -> str:
        """Get the LangGraph workflow as a mermaid diagram string.

        :returns: Mermaid diagram syntax as string
        """
        try:
            logger.info("Generating graph as mermaid diagram")
            mermaid_str = self.graph.get_graph().draw_mermaid()
            logger.info(f"✓ Generated mermaid diagram ({len(mermaid_str)} characters)")
            return mermaid_str
        except Exception as e:
            logger.error(f"Failed to generate mermaid diagram: {e}")
            raise

    def get_graph_ascii(self) -> str:
        """Get ASCII representation of the graph.

        :returns: ASCII art representation of the graph
        """
        try:
            logger.info("Generating ASCII graph representation")
            ascii_str = self.graph.get_graph().print_ascii()
            logger.info("✓ Generated ASCII graph representation")
            return ascii_str
        except Exception as e:
            logger.error(f"Failed to generate ASCII graph: {e}")
            raise
