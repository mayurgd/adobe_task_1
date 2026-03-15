"""
answer_query.py

End-to-end RAG pipeline for CFO-level document Q&A.

Pipeline
--------
1. Retrieve top-k chunks from the document's ChromaDB collection.
2. Generate a CFO-grade markdown answer via the LLM (gpt-4o by default).
3. A second LLM call identifies which [Source N] numbers were actually used.
4. Return a JSON-serialisable dict ready for frontend consumption.

Return shape
------------
{
    "query": str,
    "answer": str,                  # markdown — [Source N] refs inline
    "cited_sources": [
        {
            "number": int,          # matches [Source N] in the answer
            "heading": str,
            "pages": [int, ...],
            "text_preview": str,    # first 200 chars — use for tooltip / highlight
            "score": float
        },
        ...
    ]
}

Public API
----------
    answer_query(query, collection_name, top_k, model) -> dict
    print_result(result)                                -> None  (notebook helper)
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Allow `config` to be found when running from inside kb_setup or via notebook
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from kb_setup.retriever import query_documents

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = textwrap.dedent(
    """\
    You are a senior financial analyst and advisor to C-suite leadership.
    Your role is to synthesise document excerpts into precise, board-ready insights.

    Guidelines:
    - Write in clear, executive-level English. Be direct; avoid filler.
    - Structure your response with markdown:
        • Use **bold** for key figures and conclusions.
        • Use bullet lists only when enumerating 3+ distinct items.
        • Use markdown tables when comparing metrics or presenting structured data.
        • End every response with a "### Summary" section (2-3 sentences max).
    - Reference sources inline as [Source N] where N matches the chunk number.
    - If the context does not contain enough information, say so explicitly.
    - Never hallucinate numbers or facts not present in the context.
"""
)

_ANSWER_USER_TMPL = textwrap.dedent(
    """\
    ## Document Context

    {context}

    ---

    ## Question

    {query}

    Provide a CFO-level answer using only the context above.
    Reference relevant chunks as [Source 1], [Source 2], etc.
"""
)

_SOURCE_SYSTEM = textwrap.dedent(
    """\
    You are a citation extraction assistant.
    Given a markdown answer, extract every source number referenced (e.g. [Source 1], [Source 3]).
    Respond ONLY with a JSON array of integers, e.g. [1, 3].
    No explanation, no markdown fences, no extra text.
"""
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def answer_query(
    query: str,
    collection_name: str,
    top_k: int | None = None,
    model: Optional[str] = None,
) -> dict:
    """Run the full RAG pipeline and return a JSON-serialisable result dict.

    Args:
        query:           Natural-language question from the user.
        collection_name: ChromaDB collection name (from doc_registry).
        top_k:           Chunks to retrieve (default: settings.default_top_k).
        model:           OpenAI model override (default: settings.openai_model).

    Returns:
        {
            "query": str,
            "answer": str,           # markdown with [Source N] inline refs
            "cited_sources": [
                {
                    "number": int,
                    "heading": str,
                    "pages": [int],
                    "text_preview": str,   # first 200 chars of chunk text
                    "score": float,
                },
                ...
            ]
        }
    """
    llm_model = model or settings.openai_model
    client = OpenAI(api_key=settings.openai_api_key)

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    chunks = query_documents(query, collection_name=collection_name, top_k=top_k)
    for i, chunk in enumerate(chunks, start=1):
        chunk["source_number"] = i

    context = _build_numbered_context(chunks)

    # ── Step 2: Generate CFO-grade answer ─────────────────────────────────────
    answer_response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {
                "role": "user",
                "content": _ANSWER_USER_TMPL.format(context=context, query=query),
            },
        ],
        temperature=0.2,
    )
    answer_text = answer_response.choices[0].message.content.strip()

    # ── Step 3: Extract cited source numbers ──────────────────────────────────
    cited_numbers = _extract_cited_sources(client, llm_model, answer_text)

    # ── Step 4: Build cited_sources list (only chunks actually referenced) ────
    chunk_by_num = {c["source_number"]: c for c in chunks}
    cited_sources = [
        {
            "number": n,
            "heading": chunk_by_num[n]["heading"],
            "pages": chunk_by_num[n]["pages"],
            "text_preview": chunk_by_num[n]["text"][:200].strip(),
            "score": chunk_by_num[n]["score"],
        }
        for n in sorted(cited_numbers)
        if n in chunk_by_num
    ]

    return {
        "query": query,
        "answer": answer_text,
        "cited_sources": cited_sources,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_numbered_context(chunks: list[dict]) -> str:
    """Render chunks as a numbered context block for the LLM."""
    parts = []
    for chunk in chunks:
        n = chunk["source_number"]
        header = (
            f"### Source {n} [heading: {chunk['heading']} | pages: {chunk['pages']}]"
        )
        body_parts = []
        if chunk["text"]:
            body_parts.append(chunk["text"])
        for t in chunk.get("tables", []):
            label = "[TABLE" + (f" — {t['caption']}" if t.get("caption") else "") + "]"
            body_parts.append(f"{label}\n{t['text'] or t['html']}")
        for img in chunk.get("images", []):
            label = (
                "[IMAGE" + (f" — {img['caption']}" if img.get("caption") else "") + "]"
            )
            body_parts.append(label)
        parts.append(header + "\n\n" + "\n\n".join(body_parts))
    return "\n\n---\n\n".join(parts)


def _extract_cited_sources(client: OpenAI, model: str, answer: str) -> list[int]:
    """Second LLM call: return list of [Source N] integers cited in the answer."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SOURCE_SYSTEM},
                {"role": "user", "content": f"Answer:\n\n{answer}"},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        numbers = json.loads(raw)
        return [int(n) for n in numbers if isinstance(n, (int, float))]
    except Exception:
        # Fallback: parse [Source N] directly from the answer text
        return [int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer)]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook / CLI helpers
# ─────────────────────────────────────────────────────────────────────────────


def print_result(result: dict) -> None:
    """Pretty-print an answer_query() result dict to stdout."""
    print("=" * 72)
    print(f"QUERY : {result['query']}")
    print("=" * 72)
    print()
    print(result["answer"])
    print()
    print("─" * 72)
    sources = result["cited_sources"]
    print(f"CITED SOURCES ({len(sources)} used)")
    print("─" * 72)
    for s in sources:
        print(
            f"  [Source {s['number']}]  {s['heading']}  "
            f"|  pages {s['pages']}  |  score {s['score']}"
        )
        if s["text_preview"]:
            print(f"           {s['text_preview'][:120]}…")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG pipeline for document Q&A")
    parser.add_argument("--query", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    result = answer_query(args.query, args.collection, top_k=args.top_k)
    print_result(result)
