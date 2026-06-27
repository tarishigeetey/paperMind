'''
LLM Prompt templates for the Agentic RAG workflow.

=== JAVA COMPARISON ===
This file = a constants/templates class in Java.

Java equivalent:
    public final class AgentPrompts {
        public static final String GRADE_DOCUMENTS_PROMPT = """
            You are a grader assessing relevance...
            """;
        public static final String REWRITE_PROMPT = """ ... """;
        // ... etc
    }

WHY A SEPARATE FILE?
Same reasoning as why you never hardcode SQL inside a @Service method
in Spring — you put it in a .sql file, a @Query annotation, or a
constants class. Prompts are:
  1. The thing you'll tune MOST after deployment (based on bad outputs
     you spot in Langfuse traces)
  2. Reusable across multiple nodes
  3. Easier to A/B test when isolated from business logic

PYTHON STRING FORMATTING NOTE:
These are Python "format strings" — you'll see {question} and {context}
placeholders. They work like Java's String.format() or text blocks:

    Python: GRADE_DOCUMENTS_PROMPT.format(context=docs, question=query)
    Java:   String.format(GRADE_DOCUMENTS_PROMPT, docs, query)
            // or with named placeholders (Java 21+ text blocks + custom logic)

The triple-quoted """ string in Python = Java\'s triple-quoted text block
(introduced in Java 15+). Both let you write multi-line strings without
escape characters.
'''

# ============================================================
# PROMPT 1: Grade Documents
# ============================================================
# USED BY: grade_documents_node.py
# PURPOSE: After OpenSearch returns papers, ask the LLM: "Is this
#          paper actually relevant to what the user asked?"
#
# WHY THIS MATTERS: OpenSearch's hybrid search (BM25 + vector) can
# return results that are SEMANTICALLY similar but not actually
# USEFUL. E.g., searching "BERT fine-tuning" might also return a
# paper just because it mentions "BERT" once in passing. This LLM
# call is the quality filter that catches that.
#
# JAVA EQUIVALENT (as a constant):
#   public static final String GRADE_DOCUMENTS_PROMPT = """
#       You are a grader assessing relevance...
#       Retrieved Documents: %s
#       User Question: %s
#       ...
#       """;
#   String prompt = String.format(GRADE_DOCUMENTS_PROMPT, context, question);
#
# NOTE: "Respond in JSON format with 'binary_score'..." — this last line
# is critical. We're forcing the LLM to output structured JSON that maps
# DIRECTLY to our GradeDocuments Pydantic model from models.py. This is
# called "structured output prompting" — same concept as deserializing
# a REST response into a Java DTO, except here WE control the LLM's
# output shape via the prompt itself (or via LangChain's with_structured_output).
GRADE_DOCUMENTS_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Retrieved Documents:
{context}

User Question: {question}

If the documents contain keywords or semantic meaning related to the question, grade them as relevant.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant to the question.
Also provide brief reasoning for your decision.

Respond in JSON format with 'binary_score' (yes/no) and 'reasoning' fields."""


# ============================================================
# PROMPT 2: Rewrite Query
# ============================================================
# USED BY: rewrite_query_node.py
# PURPOSE: When grading fails (irrelevant docs), ask the LLM to
#          REPHRASE the user's vague query into something that will
#          retrieve better search results.
#
# EXAMPLE:
#   User asks: "tell me about that attention thing"
#   Rewritten: "What is the attention mechanism in transformer architectures?"
#
# WHY THIS WORKS: Vague queries produce vague embeddings, which means
# vector search returns mediocre matches. A well-formed, specific query
# produces a more precise embedding vector — better cosine similarity
# against the paper chunks in OpenSearch.
#
# JAVA EQUIVALENT: think of this like a "query normalization" step
# you might add before hitting Elasticsearch — except instead of regex
# rules, we're using an LLM's semantic understanding to expand intent.
REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for retrieving relevant documents.

Look at the initial question and try to reason about the underlying semantic intent or meaning.

Here is the initial question:
{question}

Formulate an improved question that will retrieve more relevant documents.
Provide only the improved question without any preamble or explanation."""


# ============================================================
# PROMPT 3: System Message
# ============================================================
# USED BY: agentic_rag.py (as the persistent system prompt for the whole graph)
# PURPOSE: This is the agent's "personality" and operating rules —
#          sent ONCE at t
#          context for the entire session.
#
# JAVA EQUIVALENT: imagine this as a Spring @Configuration class that
# sets the default behavior for a service — except instead of Java
# code enforcing the rule, we're relying on the LLM to FOLLOW
# natural-language instructions. This is the fundamental shift in
# AI engineering: instead of writing if/else code for every case,
# you write clear instructions and trust (and verify via Langfuse!)
# that the LLM follows them.
#
# NOTICE the explicit DO / DO NOT lists — this is a prompting best
# practice. LLMs follow binary rules ("when X, do Y") far more
# reliably than vague guidance ("use good judgment").
SYSTEM_MESSAGE = """You are an AI assistant specializing in academic research papers from arXiv.
Your domain of expertise is: Computer Science, Machine Learning, AI, and related technical research.

You have access to a tool to retrieve relevant research papers. Use this tool when:
- The user asks about specific research topics in CS/AI/ML
- The question requires knowledge from academic papers (e.g., "What are transformer architectures?")
- You need context from scientific literature (e.g., "How does BERT work?")

Do NOT use the tool when:
- The question is about general knowledge unrelated to research (e.g., "What is the meaning of dog?")
- The question is simple factual or mathematical (e.g., "what is 2+2?")
- The question is conversational, greeting, or personal
- The question is about topics outside CS/AI/ML research (e.g., cooking, history, medicine)

When you use the retrieval tool, you will receive relevant paper excerpts to help answer the question."""


# ============================================================
# PROMPT 4: Decision Prompt (alternative routing approach)
# ============================================================
# USED BY: an alternate/simpler routing strategy (binary RETRIEVE/RESPOND)
# PURPOSE: A simpler version of the guardrail — instead of a 0-100
#          score, just force a ONE-WORD answer: "RETRIEVE" or "RESPOND"
#
# WHY HAVE BOTH THIS AND GUARDRAIL_PROMPT?
# This is the simpler binary classifier. GUARDRAIL_PROMPT (below) is
# the more nuanced scored version. In production you'd typically pick
# ONE strategy — the course shows both so you understand the tradeoff:
#
#   Binary (this prompt):  Faster, simpler, but no nuance — a borderline
#                           query like "what is machine learning?" gets
#                           a hard yes/no with no visibility into WHY.
#   Scored (guardrail):    Slower (more tokens), but gives you a
#                           threshold you can TUNE (config.guhe start of every conversation, stays inardrail_threshold)
#                           without changing the prompt.
#
# JAVA EQUIVALENT: like choosing between a simple boolean feature flag
# vs. a percentage-based rollout flag (e.g., LaunchDarkly) — the scored
# version gives you a dial to turn, the binary version is just on/off.
DECISION_PROMPT = """You are an AI assistant that ONLY helps with academic research papers from arXiv in Computer Science, AI, and Machine Learning.

Question: "{question}"

Is this question about CS/AI/ML research that requires academic papers?

CRITICAL RULES:
- RETRIEVE: ONLY if the question is specifically about AI/ML/CS research topics (neural networks, algorithms, models, techniques)
- RESPOND: For EVERYTHING else (general knowledge, definitions, greetings, non-research questions)

Examples:
- "What are transformer architectures in deep learning?" -> RETRIEVE
- "Explain BERT model" -> RETRIEVE
- "What is the meaning of dog?" -> RESPOND (general dictionary definition)
- "What is a dog?" -> RESPOND (not about research)
- "Hello" -> RESPOND (greeting)
- "What is 2+2?" -> RESPOND (math, not research)

Answer with ONLY ONE WORD: "RETRIEVE" or "RESPOND"

Your answer:"""


# ============================================================
# PROMPT 5: Direct Response (no retrieval needed)
# ============================================================
# USED BY: out_of_scope_node.py
# PURPOSE: When the guardrail rejects a query (score < threshold),
#          this prompt generates a POLITE rejection message instead
#          of just returning a generic "I can't help with that."
#
# WHY BOTHER WITH AN LLM CALL FOR A REJECTION?
# You COULD just hardcode: return "Sorry, I only answer ML questions."
# But that's robotic and unhelpful. This prompt asks the LLM to:
#   1. Acknowledge what was asked
#   2. Explain WHY it's out of scope
#   3. Suggest a BETTER resource for that question
#
# This is the difference between a 400 Bad Request with no message
# vs. a well-crafted error response telling the client exactly what
# went wrong and how to fix it. Good API design principles apply to
# AI agent design too.
DIRECT_RESPONSE_PROMPT = """You are an AI assistant specializing in academic research papers from arXiv (Computer Science, AI, ML).

The following question appears to be outside the scope of academic research papers or doesn't require retrieval from research literature:

Question: {question}

Explain that this question is outside your domain of expertise (arXiv research papers in CS/AI/ML) and that you cannot answer it accurately. Be helpful by suggesting what kind of resource would be more appropriate for this question.

Answer:"""


# ============================================================
# PROMPT 6: Guardrail Validation (the scored approach — THE ONE WE USE)
# ============================================================
# USED BY: guardrail_node.py
# PURPOSE: THE most important prompt in the system. This is the
#          FIRST node every query hits. Scores 0-100 how relevant
#          the query is to ML/CS/AI research.
#
# NOTICE THE SCORE BUCKETS — this is a prompting technique called
# "few-shot scoring calibration." Instead of just saying "score 0-100",
# we give the LLM CONCRETE EXAMPLES at each score range:
#   80-100 → clear examples
#   60-79  → "potentially related" examples
#   40-59  → "borderline" examples
#   0-39   → clear rejection examples
#
# WHY THIS MATTERS: LLMs are notoriously inconsistent at producing
# calibrated numeric scores without anchoring examples. If you just
# say "score 0-100 for relevance," the LLM might score everything
# 70-90 regardless of actual relevance (a known LLM bias toward the
# "safe middle"). Giving explicit examples at each range fixes this —
# similar to how you'd write detailed Javadoc with @example tags to
# guide correct usage of an ambiguous API.
#
# This score gets parsed into our GuardrailScoring Pydantic model
# (from models.py), then compared against config.guardrail_threshold (60).
GUARDRAIL_PROMPT = """You are a guardrail evaluator assessing whether a user query is within the scope of academic research papers from arXiv in Computer Science, AI, and Machine Learning.

User Query: {question}

Evaluate whether this query is:
- About CS/AI/ML research topics (neural networks, algorithms, models, architectures, techniques, etc.)
- Requires academic paper knowledge to answer
- Within the domain of Computer Science research

Assign a relevance score (0-100):
- 80-100: Clearly about CS/AI/ML research (e.g., "What are transformer architectures?", "How does BERT work?")
- 60-79: Potentially research-related but unclear (e.g., "Tell me about attention mechanisms")
- 40-59: Borderline or ambiguous (e.g., "What is machine learning?")
- 0-39: NOT about research papers (e.g., "What is a dog?", "Hello", "What is 2+2?")

Provide:
1. A score between 0 and 100
2. A brief reason explaining why you gave this score

Respond in JSON format with 'score' (integer 0-100) and 'reason' (string) fields."""


# ============================================================
# PROMPT 7: Generate Answer (the final step)
# ============================================================
# USED BY: generate_answer_node.py
# PURPOSE: The LAST node in the happy path. Takes the relevant,
#          graded papers and synthesizes a final answer.
#
# THIS IS RAG'S CORE PRINCIPLE IN ACTION — notice the explicit
# constraint: "using ONLY the information from the retrieved papers"
# and "Do NOT make up information." This is how you fight LLM
# hallucination: ground every claim in retrieved, verifiable context,
# and instruct the model to refuse rather than fabricate.
#
# JAVA EQUIVALENT: think of {context} as a dependency you're injecting
# into a template — like populating a Thymeleaf/JSP template with
# data fetched from a repository, except here the "template engine"
# is the LLM, and instead of just substituting variables, it's
# reasoning over them to produce a coherent natural-language answer.
#
# "Cite specific papers... use paper titles or arxiv IDs" — this
# instruction is what gives you the SourceItem list in the final
# API response. The LLM's citations get cross-referenced against
# state.relevant_sources to build the structured response.
GENERATE_ANSWER_PROMPT = """You are an AI research assistant specializing in academic papers from arXiv in Computer Science, AI, and Machine Learning.

Your task is to answer the user's question using ONLY the information from the retrieved research papers provided below.

Retrieved Research Papers:
{context}

User Question: {question}

Instructions:
- Provide a comprehensive, accurate answer based ONLY on the retrieved papers
- Cite specific papers when making claims (use paper titles or arxiv IDs)
- If the papers don't contain enough information to fully answer the question, acknowledge this
- Structure your answer clearly and professionally
- Focus on the key insights and findings from the papers
- Do NOT make up information or cite papers not in the retrieved context

Answer:"""
