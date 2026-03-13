"""
tools.py

LangChain tool definitions exposed to the deep agent.

Each tool is a single-responsibility function decorated with ``@tool``.
Tools import only from the retriever and tracing layers — no agent-level
concerns leak in here.

Public API:
    answer_from_documents   — RAG tool: retrieve + generate.
    TOOLS                   — List of all tool instances ready to pass to
                              ``create_deep_agent(tools=TOOLS)``.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import settings
from retriever import format_chunks_as_context, query_documents
from tracing import create_generation_span, create_retrieval_span, get_langfuse

# ── Lazy Langfuse client (shared across tool invocations in a process) ────────
# The agent turn already initialises a Langfuse client; we grab the same
# singleton here so nested observations link up correctly in the trace tree.
_langfuse = None


def _lf():
    """Return the Langfuse client singleton, initialising it once if needed."""
    global _langfuse
    if _langfuse is None:
        _langfuse = get_langfuse()
    return _langfuse


# ─────────────────────────────────────────────────────────────────────────────
# RAG tool
# ─────────────────────────────────────────────────────────────────────────────

_RAG_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Using ONLY the document excerpts "
    "provided below, answer the question. Be concise and cite specific figures "
    "and page numbers where available. If the answer cannot be determined from "
    "the excerpts, say so explicitly — do not hallucinate."
)


@tool
def answer_from_documents(query: str) -> str:
    """Retrieve relevant passages from the indexed company documents and generate
    a factual, source-grounded answer.

    Use this tool whenever the user asks about specific figures, financial results,
    strategies, risks, or any content that may be found in the uploaded company
    documents (annual reports, quarterly reports, etc.).

    Args:
        query: The user's question expressed in natural language.

    Returns:
        A concise answer grounded in the retrieved document excerpts, with
        references to headings and page numbers where available.
    """
    lf = _lf()

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    try:
        with create_retrieval_span(lf, query, settings.default_top_k) as span:
            chunks = query_documents(query, top_k=settings.default_top_k)
            span.update(
                output={
                    "chunks_retrieved": len(chunks),
                    "headings": [c.get("heading", "") for c in chunks],
                }
            )
    except Exception as exc:
        return f"[retrieval error] Could not query the document index: {exc}"

    if not chunks:
        return "No relevant content found in the document index for this query."

    context = format_chunks_as_context(chunks)
    rag_prompt = (
        f"{_RAG_SYSTEM_PROMPT}\n\n"
        f"Question: {query}\n\n"
        f"Document excerpts:\n{context}"
    )

    # ── Stage 2: Generation ───────────────────────────────────────────────────
    model_name = settings.openai_model
    llm = ChatOpenAI(model=model_name, temperature=0)

    with create_generation_span(lf, model_name, rag_prompt) as gen:
        response = llm.invoke(rag_prompt)
        gen.update(output=response.content)

    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — pass this to create_deep_agent
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list = [answer_from_documents]
