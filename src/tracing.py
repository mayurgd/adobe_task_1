"""
tracing.py

Langfuse observability helpers.

Provides a thin, optional wrapper around the Langfuse ``get_client()`` API so
that the rest of the codebase can call ``get_langfuse()`` and get either a live
client or ``None`` — without scattering ``try/except ImportError`` blocks
everywhere.

When Langfuse credentials are not configured (or the package is not installed),
all functions degrade gracefully so the agent works without any tracing.

Public API:
    get_langfuse()                          -> Langfuse | None
    create_retrieval_span(lf, query, top_k) -> context manager | null context
    create_generation_span(lf, model, prompt) -> context manager | null context
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from config import settings

if TYPE_CHECKING:
    # Only used for type hints — not imported at runtime to keep tracing optional
    from langfuse import Langfuse  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────────────


def get_langfuse() -> "Langfuse | None":
    """Return a configured Langfuse client, or ``None`` if tracing is disabled.

    Tracing is disabled when:
      - ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` env vars are absent.
      - The ``langfuse`` package is not installed.

    The client reads credentials from the environment automatically via
    ``get_client()`` (langfuse >= 2.x).

    Returns:
        A live ``Langfuse`` client, or ``None``.
    """
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        print(f"[info] Langfuse tracing enabled → {settings.langfuse_host}")
        return client

    except ImportError:
        print(
            "[warn] langfuse package not found — tracing disabled.\n"
            "       Install it with:  pip install langfuse"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers — each returns a context manager (real or null)
# ─────────────────────────────────────────────────────────────────name────────


@contextlib.contextmanager
def _null_ctx(**_kwargs: Any):
    """A no-op context manager used when Langfuse is unavailable."""

    class _Stub:
        def update(self, **__: Any) -> None:
            pass

    yield _Stub()


def create_retrieval_span(
    langfuse: "Langfuse | None",
    query: str,
    top_k: int,
):
    """Context manager that wraps a document-retrieval step in a Langfuse span.

    Usage::

        with create_retrieval_span(lf, query, top_k) as span:
            chunks = query_documents(query, top_k)
            span.update(output={"chunks_retrieved": len(chunks)})

    When *langfuse* is ``None`` the context manager is a silent no-op.

    Args:
        langfuse: Live Langfuse client, or ``None``.
        query:    The search query string.
        top_k:    Number of chunks requested.

    Yields:
        A Langfuse span observation (or a stub with a no-op ``update()``).
    """
    if langfuse is None:
        return _null_ctx()

    return langfuse.start_as_current_observation(
        as_type="span",
        name="document-retrieval",
        input={"query": query, "top_k": top_k},
    )


def create_generation_span(
    langfuse: "Langfuse | None",
    model_name: str,
    prompt: str,
):
    """Context manager that wraps an LLM generation step in a Langfuse observation.

    Usage::

        with create_generation_span(lf, model_name, rag_prompt) as gen:
            response = llm.invoke(rag_prompt)
            gen.update(output=response.content)

    When *langfuse* is ``None`` the context manager is a silent no-op.

    Args:
        langfuse:    Live Langfuse client, or ``None``.
        model_name:  Name of the LLM model being called (e.g. ``"gpt-4o-mini"``).
        prompt:      The full prompt string passed to the LLM.

    Yields:
        A Langfuse generation observation (or a stub with a no-op ``update()``).
    """
    if langfuse is None:
        return _null_ctx()

    return langfuse.start_as_current_observation(
        as_type="generation",
        name="rag-generation",
        model=model_name,
        input=prompt,
    )


def create_agent_turn_span(
    langfuse: "Langfuse | None",
    user_input: str,
):
    """Context manager that wraps a full agent turn in a Langfuse span.

    Nests an inner ``agent-invocation`` span automatically.

    Usage::

        with create_agent_turn_span(lf, user_input) as span:
            reply = await _stream_agent_turn(agent, conversation)
            span.update(output=reply)

    When *langfuse* is ``None`` the context manager is a silent no-op.
    """
    if langfuse is None:
        return _null_ctx()

    return langfuse.start_as_current_observation(
        as_type="span",
        name="agent-turn",
        input=user_input,
    )
