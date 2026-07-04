# src/services/langfuse/client.py
#
# Langfuse v4 tracing client (SDK rewritten March 2026)
#
# BREAKING CHANGE from v2/v3:
#   REMOVED: Langfuse().trace(), Langfuse().span(), Langfuse().generation()
#   NEW API:  get_client() + start_as_current_observation()
#
# Java analogy: like Spring Boot 2 → 3 migration.
# Same concepts (spans, traces), completely new method names.
#
# v4 pattern:
#   langfuse = get_client()   # singleton, reads env vars automatically
#   with langfuse.start_as_current_observation(as_type="span", name="x") as span:
#       span.update(input=..., output=...)
#       # nested observations are automatically children

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from langfuse import get_client
from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Langfuse v4 tracing wrapper."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                # v4: get_client() reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
                # LANGFUSE_HOST from env automatically.
                # Java analogy: DataSource.getConnection() from connection pool
                import os

                os.environ["LANGFUSE_PUBLIC_KEY"] = self.settings.public_key
                os.environ["LANGFUSE_SECRET_KEY"] = self.settings.secret_key
                os.environ["LANGFUSE_HOST"] = self.settings.host

                self.client = get_client()
                logger.info(f"Langfuse v4 tracing initialized (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Top-level trace for /ask endpoint.

        v4 API: start_as_current_observation() creates the root span.
        All child spans created inside this block are automatically nested.

        Java analogy: try-with-resources transaction boundary
        """
        if not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name="rag_request",
                input={"query": query},
                metadata={
                    **(metadata or {}),
                    "user_id": user_id,
                    "session_id": session_id,
                },
            ) as span:
                # Set trace-level attributes for filtering in Langfuse UI
                self.client.update_current_trace(
                    user_id=user_id,
                    session_id=session_id,
                    tags=["rag", "ask"],
                )
                yield span
        except Exception as e:
            logger.error(f"Error creating Langfuse trace: {e}")
            yield None
        finally:
            self.flush()

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a child span.

        NOTE: In v4, spans are best created with context managers, not manually.
        This method exists for compatibility with RAGTracer which calls it directly.
        Returns a no-op object when called outside a trace context.

        Java analogy: creating a savepoint inside an existing transaction
        """
        if not self.client:
            return None

        try:
            # v4: use start_as_current_observation as a regular (non-context-manager) call
            # We store the span object and the caller calls span.end() manually
            span = self.client.start_as_current_span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
            return span
        except AttributeError:
            # Fallback: start_as_current_span may not exist in all v4 builds
            # Return a no-op span so callers don't crash
            return _NoOpSpan(name)
        except Exception as e:
            logger.error(f"Error creating span {name}: {e}")
            return _NoOpSpan(name)

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """Update a span with output or metadata."""
        if not span or isinstance(span, _NoOpSpan):
            return

        try:
            update_data = {}
            if output is not None:
                update_data["output"] = output
            if metadata:
                update_data["metadata"] = metadata
            if level:
                update_data["level"] = level
            if status_message:
                update_data["status_message"] = status_message
            if update_data:
                span.update(**update_data)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    def end_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """End a span with optional final output."""
        if not span or isinstance(span, _NoOpSpan):
            return

        try:
            if output is not None or metadata is not None:
                self.update_span(span, output=output, metadata=metadata)
            span.end()
        except Exception as e:
            logger.error(f"Error ending span: {e}")

    def score_trace(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Score the current trace."""
        if not self.client:
            return

        try:
            # v4: score_current_trace() when inside a trace context
            self.client.score_current_trace(
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Error scoring trace: {e}")

    def get_callback_handler(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Get LangChain CallbackHandler for /agentic-ask LangGraph integration.
        Pass to graph.invoke(config={"callbacks": [handler]})
        """
        if not self.client:
            return None

        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler()
        except Exception as e:
            logger.error(f"Error creating CallbackHandler: {e}")
            return None

    @contextmanager
    def trace_langgraph_agent(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Context manager for /agentic-ask LangGraph tracing."""
        if not self.client:
            yield (None, None)
            return

        handler = self.get_callback_handler(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )
        yield (None, handler)

    def flush(self):
        """Flush pending traces to Langfuse Cloud."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Flush and shutdown on app teardown."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")


class _NoOpSpan:
    """
    Fallback no-op span — absorbs all calls silently.
    Java analogy: NullObject pattern — prevents NullPointerException
    when tracing is unavailable.
    """

    def __init__(self, name: str = ""):
        self.name = name

    def update(self, **kwargs):
        pass

    def end(self):
        pass
