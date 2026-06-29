"""
The Verify Citations Node — Strategy B: detect AND correct.

=== WHY STRATEGY B OVER STRATEGY A ===
Strategy A (annotate) just adds a disclaimer next to a wrong citation —
honest, but the user still has to do the work of figuring out what's
actually true. Strategy B spends one more LLM call to ask the model to
fix its own mistake, using ONLY the real retrieved context as ground
truth. This is the stronger answer for a production system: don't just
detect a problem, resolve it when you reasonably can.

=== JAVA COMPARISON ===
This is the same shape as a RETRY-WITH-CORRECTED-INPUT pattern you'd
write for a flaky external call that sometimes returns malformed data:

    public ValidatedResponse processWithCorrection(String rawAnswer, Context ctx) {
        ValidationResult result = validate(rawAnswer);
        if (result.isValid()) {
            return new ValidatedResponse(rawAnswer, false);
        }
        // one corrective retry, NOT an infinite loop
        String corrected = correctionService.regenerate(rawAnswer, result.getErrors(), ctx);
        return new ValidatedResponse(corrected, true);
    }

Exactly one retry, bounded — same discipline as retrieve_node's
max_retrieval_attempts guard from Episode 6. We never loop verification
indefinitely; if the corrected answer STILL has issues, we accept it
and move on rather than spending unbounded LLM calls chasing perfection.
"""

import logging
import re
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)


def _extract_cited_papers(answer_text: str) -> List[str]:
    """Pull out anything that looks like a citation reference:
    "Vaswani et al., 2017", "Dai et al. (2020)", etc.
    """
    pattern = r"([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?),?\s*\(?(\d{4})\)?"
    matches = re.findall(pattern, answer_text)
    seen = list(dict.fromkeys(f"{author} {year}" for author, year in matches))
    return seen


def _build_source_fingerprint(relevant_sources: list, raw_context: str) -> str:
    """Build a single searchable blob of everything we KNOW is real."""
    parts = [raw_context.lower()]
    for source in relevant_sources:
        title = getattr(source, "title", "") or ""
        authors = getattr(source, "authors", []) or []
        parts.append(title.lower())
        parts.append(" ".join(authors).lower())
    return " ".join(parts)


def _find_unverified_citations(cited_papers: List[str], fingerprint: str) -> List[str]:
    """Check each cited paper's author name against the real source
    fingerprint. Returns the ones that don't appear anywhere in the
    actual retrieved content.
    """
    unverified = []
    for citation in cited_papers:
        author_part = citation.split()[0].lower()
        if author_part not in fingerprint:
            unverified.append(citation)
    return unverified


async def _regenerate_with_correction(
    answer_text: str,
    unverified: List[str],
    question: str,
    context: str,
    runtime: Runtime[Context],
) -> str:
    """Send the flawed answer back to the LLM with explicit instructions
    to fix or remove the unverified citations, using ONLY the real
    retrieved context as ground truth.

    :param answer_text: The original (possibly hallucinated) answer
    :param unverified: List of citation strings that failed verification
    :param question: The original user question, for context
    :param context: The actual retrieved paper text — ground truth
    :param runtime: Runtime context (for the Ollama client + model name)
    :returns: The corrected answer text

    This costs exactly ONE additional LLM call. We use temperature=0.0
    here too — correction is a precision task, not a creative one, same
    reasoning as guardrail/grading nodes from earlier episodes.
    """
    correction_prompt = f"""You previously answered a question, but some citations in your answer could not be verified against the actual source material.

Original question: {question}

Your previous answer:
{answer_text}

The following citations could NOT be verified against the retrieved papers: {", ".join(unverified)}

Actual retrieved source material (this is the ONLY ground truth you may cite from):
{context}

Rewrite your answer using ONLY information and citations that appear in the source material above. If a claim cannot be supported by this source material, remove it entirely rather than guessing or relying on your own prior knowledge. If you cannot verify a citation's author or year from the source material, omit the citation rather than inventing one.

Corrected answer:"""

    llm = runtime.context.ollama_client.get_langchain_model(
        model=runtime.context.model_name,
        temperature=0.0,
    )

    logger.info(f"Regenerating answer to correct {len(unverified)} unverified citation(s)")
    corrected = await llm.ainvoke(correction_prompt)

    corrected_text = corrected.content if hasattr(corrected, "content") else str(corrected)
    return corrected_text


async def ainvoke_verify_citations_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Check the generated answer's citations against what was ACTUALLY
    retrieved. If anything looks fabricated, regenerate a corrected
    answer using only verified ground truth (Strategy B).

    :param state: Current agent state (must run AFTER generate_answer)
    :param runtime: Runtime context
    :returns: Dictionary with messages — either unchanged, or replaced
              with a corrected, regenerated answer

    Cost note: this node is FREE (no LLM call) when no citations are
    found, or when all citations verify cleanly. It costs exactly ONE
    extra LLM call only in the case where correction is actually
    needed — the common case stays cheap.
    """
    logger.info("NODE: verify_citations")

    messages = state["messages"]
    if not messages:
        return {}

    last_message = messages[-1]
    answer_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    relevant_sources = state.get("relevant_sources", [])

    from .utils import get_latest_context, get_latest_query

    raw_context = get_latest_context(messages) or ""
    question = get_latest_query(messages)

    cited_papers = _extract_cited_papers(answer_text)

    if not cited_papers:
        logger.debug("No citation-like patterns found in answer, skipping verification")
        return {}

    fingerprint = _build_source_fingerprint(relevant_sources, raw_context)
    unverified = _find_unverified_citations(cited_papers, fingerprint)

    if not unverified:
        logger.info("All citations verified against retrieved context")
        return {}

    logger.warning(f"Unverified citations detected: {unverified}, regenerating corrected answer")

    try:
        corrected_text = await _regenerate_with_correction(
            answer_text=answer_text,
            unverified=unverified,
            question=question,
            context=raw_context,
            runtime=runtime,
        )

        # === Bounded re-check: did correction actually work? ===
        # We check ONCE more, but we do NOT loop back into correction
        # again if it's still imperfect — that would risk an unbounded
        # cycle of LLM calls. Same discipline as retrieve_node's
        # max_retrieval_attempts guard: try the corrective step exactly
        # once, then accept the result either way.
        still_unverified = _find_unverified_citations(_extract_cited_papers(corrected_text), fingerprint)

        if still_unverified:
            # Correction didn't fully resolve it — be transparent
            # rather than silently shipping a still-flawed answer.
            logger.warning(
                f"Correction pass still has unverified citations: {still_unverified}, "
                "appending disclaimer rather than retrying again"
            )
            corrected_text += (
                "\n\n⚠️ **Note:** Some citations in this answer could not be fully "
                f"verified against the retrieved papers: {', '.join(still_unverified)}. "
                "Please verify against original sources before relying on them."
            )
        else:
            logger.info("Correction successful — all citations now verified")

        return {"messages": [AIMessage(content=corrected_text)]}

    # === Fallback — same circuit-breaker pattern as every other node ===
    # If the correction LLM call itself fails (Ollama down, timeout,
    # etc.), don't crash the whole request over a verification step.
    # Fall back to Strategy A's behavior: annotate the ORIGINAL answer
    # with a disclaimer instead of losing the answer entirely.
    except Exception as e:
        logger.error(f"Citation correction failed: {e}, falling back to disclaimer annotation")
        disclaimer = (
            "\n\n⚠️ **Note:** This answer references citation(s) that could not be "
            f"verified against the retrieved papers: {', '.join(unverified)}. "
            "These may be inaccurate — please verify against the original sources before relying on them."
        )
        return {"messages": [AIMessage(content=answer_text + disclaimer)]}
