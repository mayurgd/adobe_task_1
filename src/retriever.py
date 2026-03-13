"""
retriever.py

Query-time retrieval: semantic search over the ChromaDB collection, two-stage
asset hydration, and context serialisation for LLM consumption.

Two-stage retrieval
-------------------
Stage 1 — Semantic search
    The query is embedded with the same model used at index time.  ChromaDB
    returns the top-k most similar chunk IDs along with their compact metadata.

Stage 2 — Asset hydration
    Table HTML and image paths were intentionally NOT stored in ChromaDB
    metadata (too bulky).  Instead, each chunk carries a ``source_idx``
    pointer back to the original ``content_list.json``.  The hydration step
    looks up that index and attaches the full asset to the result dict.

Lazy singletons
---------------
The embedding model, ChromaDB collection, and content list are each loaded once
on first use and cached for the lifetime of the process.  This avoids repeated
cold-start costs across multi-turn agent conversations.

Public API:
    query_documents(query, top_k)  -> list[dict]
    format_chunks_as_context(chunks) -> str
    retrieve(query, collection, model, content, top_k) -> list[dict]   # testable
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

ContentList = list[dict[str, Any]]
ResultChunk = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Core retrieval (pure — no singletons, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    content: ContentList,
    top_k: int | None = None,
) -> list[ResultChunk]:
    """Perform a two-stage semantic retrieval for *query*.

    Args:
        query:      Natural-language question to search for.
        collection: ChromaDB collection created by ``indexer.build_index()``.
        model:      SentenceTransformer instance (same model used at index time).
        content:    Parsed ``content_list.json`` used for Stage 2 hydration.
        top_k:      Number of results to return.  Defaults to
                    ``settings.default_top_k``.

    Returns:
        List of result dicts (ranked by cosine similarity, best first):

        ============  ===================================================
        Key           Description
        ============  ===================================================
        rank          1-based rank (int).
        score         Cosine similarity in [0, 1] (float).
        type          ``"text"`` | ``"table"`` | ``"image"``
        heading       Section heading above this chunk.
        pages         List of page indices (int).
        text          Plain-text body (paragraph, stringified table, caption).
        table_html    Full HTML of the table (table chunks only; else ``""``)
        img_path      Relative image path (image chunks only; else ``""``)
        caption       Caption string (table / image chunks).
        ============  ===================================================
    """
    k = top_k if top_k is not None else settings.default_top_k

    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )

    output: list[ResultChunk] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Stage 2: hydrate full asset from the original content_list
        source_idx = meta.get("source_idx", -1)
        table_html = ""
        img_path = meta.get("img_path", "")

        if 0 <= source_idx < len(content):
            raw = content[source_idx]
            if meta["type"] == "table":
                table_html = raw.get("table_body", "")
            elif meta["type"] == "image":
                img_path = raw.get("img_path", img_path)

        output.append(
            {
                "rank": i + 1,
                "score": round(1 - distance, 4),
                "type": meta["type"],
                "heading": meta["heading"],
                "pages": json.loads(meta["pages"]),
                "text": meta["text"],
                "table_html": table_html,
                "img_path": img_path,
                "caption": meta["caption"],
            }
        )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Context serialiser
# ─────────────────────────────────────────────────────────────────────────────


def format_chunks_as_context(chunks: list[ResultChunk]) -> str:
    """Serialise retrieved chunks into a labelled context block for an LLM.

    Each chunk is rendered as::

        [TYPE | heading: … | pages: … | score: …]
        <body text>

    Chunks are separated by ``---`` dividers so the LLM can clearly attribute
    each piece of evidence to its source.

    Args:
        chunks: List of result dicts as returned by ``retrieve()`` or
                ``query_documents()``.

    Returns:
        A single string ready to be inserted into an LLM prompt.
    """
    parts: list[str] = []
    for chunk in chunks:
        header = (
            f"[{chunk['type'].upper()} | heading: {chunk['heading']} "
            f"| pages: {chunk['pages']} | score: {chunk['score']}]"
        )
        if chunk["type"] == "text":
            body = chunk["text"]
        elif chunk["type"] == "table":
            body = chunk["text"] or chunk["table_html"]
        else:  # image
            body = chunk["caption"] or "(image — no text available)"

        parts.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singletons — shared across the agent's multi-turn session
# ─────────────────────────────────────────────────────────────────────────────

_singleton_collection: chromadb.Collection | None = None
_singleton_model: SentenceTransformer | None = None
_singleton_content: ContentList | None = None


def get_retrieval_resources() -> tuple[chromadb.Collection, SentenceTransformer, ContentList]:
    """Return ``(collection, model, content_list)``, initialising lazily on first call.

    Subsequent calls return the cached instances instantly.  This pattern avoids
    repeated model-loading cost across agent turns.

    Returns:
        3-tuple of (ChromaDB collection, SentenceTransformer, content list).

    Raises:
        chromadb.errors.InvalidCollectionException: If the ChromaDB collection
            has not been built yet (run ``index_documents.py`` first).
    """
    global _singleton_collection, _singleton_model, _singleton_content

    if _singleton_model is None:
        print(f"[info] Loading embedding model {settings.embedding_model} …")
        _singleton_model = SentenceTransformer(settings.embedding_model)

    if _singleton_collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        _singleton_collection = client.get_collection(settings.collection_name)
        print(f"[info] Connected to ChromaDB collection '{settings.collection_name}'")

    if _singleton_content is None and settings.content_list_path.exists():
        import json as _json

        with open(settings.content_list_path, encoding="utf-8") as fh:
            _singleton_content = _json.load(fh)

    return _singleton_collection, _singleton_model, _singleton_content or []


def query_documents(query: str, top_k: int | None = None) -> list[ResultChunk]:
    """Embed *query* and return the top-k most relevant chunks.

    This is the primary entry point used by ``tools.py``.  It uses the lazy
    singletons so the first call pays the model-load cost; subsequent calls
    are fast.

    Args:
        query:  Natural-language question.
        top_k:  Number of results.  Defaults to ``settings.default_top_k``.

    Returns:
        List of result dicts as described in ``retrieve()``.
    """
    collection, model, content = get_retrieval_resources()
    return retrieve(query, collection, model, content, top_k=top_k)
