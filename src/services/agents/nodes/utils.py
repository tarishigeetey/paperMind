"""
Shared helper functions used across multiple LangGraph nodes.

=== JAVA COMPARISON ===
This file = a static utility class, exactly like:

    public final class MessageUtils {
        private MessageUtils() {}  // prevent instantiation

        public static List<SourceItem> extractSourcesFromToolMessages(List<Message> messages) { ... }
        public static List<ToolArtefact> extractToolArtefacts(List<Message> messages) { ... }
        public static ReasoningStep createReasoningStep(String stepName, String description, Map<String,Object> metadata) { ... }
        public static List<Message> filterMessages(List<Message> messages) { ... }
        public static String getLatestQuery(List<Message> messages) { ... }
        public static String getLatestContext(List<Message> messages) { ... }
    }

WHY EXTRACT THESE INTO A SEPARATE FILE?
Every node (guardrail, retrieve, grade, generate) needs to answer the
SAME question: "what did the user actually ask?" Without this utils
file, you'd copy-paste the same for-loop into 4+ node files. Classic
DRY principle — same reason you'd pull common logic into a shared
@Component or static utility class in Spring rather than duplicating
it across @Service classes.

PYTHON NOTE: Python modules (files) ARE namespaces by default — there's
no need for a "private constructor" trick like Java's utility classes.
You just write top-level functions; the file itself acts as the
namespace. So `from .utils import get_latest_query` is the Python
equivalent of `import static com.papermind.MessageUtils.getLatestQuery;`
"""

import logging
from typing import Dict, List, Optional

# These are LangChain's message type classes — think of them as a
# small class hierarchy:
#   Java: interface Message { ... }
#         class HumanMessage implements Message { ... }   // user said this
#         class AIMessage implements Message { ... }       // LLM said this
#         class ToolMessage implements Message { ... }      // tool result
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Our domain models from models.py
from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)
# Java: private static final Logger logger = LoggerFactory.getLogger(Utils.class);


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract source papers from ToolMessages in the conversation.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects

    JAVA EQUIVALENT:
        public static List<SourceItem> extractSourcesFromToolMessages(List<Message> messages) {
            List<SourceItem> sources = new ArrayList<>();
            for (Message msg : messages) {
                if (msg instanceof ToolMessage tm && "retrieve_papers".equals(tm.getName())) {
                    // parse tool response for sources
                }
            }
            return sources;
        }

    NOTE: This is currently a STUB (returns empty list) in the course
    code — it's a placeholder for parsing actual paper metadata out of
    the raw tool response JSON. Same as seeing a Java method with a
    // TODO: implement actual parsing comment — functional skeleton,
    not yet wired to real data.
    """
    sources = []

    # isinstance() check = Java's "instanceof" pattern matching
    # Java 17+: if (msg instanceof ToolMessage tm) { ... }
    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            if msg.name == "retrieve_papers":
                # Parse tool response for sources
                # This would need to parse the actual document metadata
                # For now, return empty list
                pass

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool execution artifacts (for debugging/tracing) from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects

    JAVA EQUIVALENT:
        public static List<ToolArtefact> extractToolArtefacts(List<Message> messages) {
            List<ToolArtefact> artefacts = new ArrayList<>();
            for (Message msg : messages) {
                if (msg instanceof ToolMessage tm) {
                    artefacts.add(new ToolArtefact(
                        tm.getName() != null ? tm.getName() : "unknown",
                        tm.getToolCallId() != null ? tm.getToolCallId() : "",
                        tm.getContent(),
                        new HashMap<>()
                    ));
                }
            }
            return artefacts;
        }
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # getattr(obj, "attr", default) = Java's safe navigation +
            # default fallback in one call:
            #   Java: msg.getName() != null ? msg.getName() : "unknown"
            #   Python: getattr(msg, "name", "unknown")
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record — a single entry in the agent's
    "explain yourself" trail shown to the user.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object

    JAVA EQUIVALENT:
        public static ReasoningStep createReasoningStep(
                String stepName, String description, Map<String, Object> metadata) {
            return new ReasoningStep(stepName, description,
                metadata != null ? metadata : new HashMap<>());
        }

    PYTHON NOTE: `metadata or {}` is a common Python idiom.
    If metadata is None (falsy), it evaluates to {} instead.
    Java: metadata != null ? metadata : new HashMap<>();
    Same ternary-with-null-check pattern, just terser syntax.
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes ToolMessage and SystemMessage — useful when building a
    clean "conversation view" for the user that hides internal tool
    chatter (search calls, retrieval results).

    :param messages: List of messages to filter
    :returns: Filtered list of messages

    JAVA EQUIVALENT:
        public static List<Message> filterMessages(List<Message> messages) {
            return messages.stream()
                .filter(msg -> msg instanceof HumanMessage || msg instanceof AIMessage)
                .collect(Collectors.toList());
        }

    NOTE: `AIMessage | HumanMessage` in the return type is Python's
    UNION TYPE syntax (added in Python 3.10+) — same as Java's planned
    sealed interface pattern, or simply "this returns either type."
    """
    # This is a Python LIST COMPREHENSION — the single most common
    # Python idiom you'll see everywhere. It's equivalent to:
    #   Java: messages.stream()
    #           .filter(msg -> msg instanceof HumanMessage || msg instanceof AIMessage)
    #           .collect(Collectors.toList());
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the most recent user query from the message history.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found

    JAVA EQUIVALENT:
        public static String getLatestQuery(List<Message> messages) {
            for (int i = messages.size() - 1; i >= 0; i--) {
                if (messages.get(i) instanceof HumanMessage hm) {
                    return hm.getContent();
                }
            }
            throw new IllegalStateException("No user query found in messages");
        }

    WHY SEARCH IN REVERSE (reversed())?
    AgentState.messages is APPEND-ONLY (remember add_messages reducer
    from state.py episode). The conversation grows over time:
        [HumanMsg, AIMsg(tool_call), ToolMsg, AIMsg(answer), HumanMsg(follow-up)]
    We want the MOST RECENT human message — i.e., what the user is
    asking RIGHT NOW, not their first message from 10 turns ago.
    Searching backwards finds it in O(1) in the common case (last msg)
    instead of scanning the whole list forward and overwriting as you go.

    raise ValueError(...) = Java's throw new IllegalArgumentException(...)
    Both are "fail fast and loud" — if there's no query, something is
    fundamentally broken upstream, so don't silently return "".
    """
    # reversed(messages) = Java's Collections.reverse() but as a lazy
    # iterator (doesn't actually mutate or copy the list)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest tool result (retrieved paper content) from messages.

    :param messages: List of messages
    :returns: Latest context text or empty string

    JAVA EQUIVALENT:
        public static String getLatestContext(List<Message> messages) {
            for (int i = messages.size() - 1; i >= 0; i--) {
                if (messages.get(i) instanceof ToolMessage tm) {
                    return tm.getContent() != null ? tm.getContent() : "";
                }
            }
            return "";
        }

    NOTE THE DIFFERENT FAILURE MODE vs get_latest_query():
    get_latest_query() RAISES an exception if missing (a query should
    ALWAYS exist — it's a bug if not).
    get_latest_context() RETURNS "" if missing (context is OPTIONAL —
    e.g., the out_of_scope path never retrieves anything, so no
    ToolMessage will exist, and that's a perfectly normal, expected case).

    This distinction — when to raise vs. when to return a safe default —
    is the same judgment call you make in Java between throwing a
    checked exception vs. returning Optional.empty() / a sentinel value.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if hasattr(msg, "content") else ""
            return content if isinstance(content, str) else str(content)

    return ""
