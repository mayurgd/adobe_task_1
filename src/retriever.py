"""
retriever.py

Query-time retrieval: semantic search over the ChromaDB collection, two-stage
asset hydration, cross-encoder reranking, and context serialisation for LLM
consumption.

Retrieval pipeline
------------------
Stage 1 — Semantic search (bi-encoder)
    The query is embedded with BGE and ChromaDB returns the top
    ``retrieval_candidate_k`` candidates (default 20) by cosine similarity.
    Casting a wider net here gives the reranker more to work with.

Stage 2 — Asset hydration
    Table HTML and image paths are looked up from the original content_list
    via each chunk's ``source_idx`` pointer. Text chunks need no hydration —
    their full text is stored directly in ChromaDB metadata.

Stage 3 — Cross-encoder reranking
    A cross-encoder model scores every (query, chunk_text) pair jointly,
    replacing the bi-encoder's approximate cosine ranking with a much more
    accurate relevance score. The top ``top_k`` results are returned.

    Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    — small (22 M params), fast on CPU, strong on English passage ranking.

Public API:
    query_documents(query, top_k)    -> list[dict]
    format_chunks_as_context(chunks) -> str
    retrieve(query, collection, model, content, top_k) -> list[dict]
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import settings

ContentList = list[dict[str, Any]]
ResultChunk = dict[str, Any]

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────────────────────────────────────
# Core retrieval (pure — no singletons, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    content: ContentList,
    top_k: int | None = None,
    reranker: CrossEncoder | None = None,
    candidate_k: int | None = None,
) -> list[ResultChunk]:
    """Perform a three-stage retrieval for *query*.

    Stage 1: bi-encoder semantic search over ChromaDB (fetches ``candidate_k``).
    Stage 2: asset hydration (table HTML / image paths from content_list).
    Stage 3: cross-encoder reranking, then truncate to ``top_k``.

    Args:
        query:       Natural-language question to search for.
        collection:  ChromaDB collection created by ``indexer.build_index()``.
        model:       SentenceTransformer instance (BGE bi-encoder).
        content:     Parsed ``content_list.json`` for Stage 2 hydration.
        top_k:       Final number of results to return after reranking.
                     Defaults to ``settings.default_top_k``.
        reranker:    Optional pre-loaded CrossEncoder. If None, reranking is
                     skipped and ``candidate_k`` results are returned as-is.
        candidate_k: How many candidates to fetch from ChromaDB before
                     reranking. Defaults to ``settings.retrieval_candidate_k``.
                     Clamped to at least ``top_k``.

    Returns:
        List of result dicts (ranked best-first):

        ============  ===================================================
        Key           Description
        ============  ===================================================
        rank          1-based rank (int).
        score         Reranker logit if reranked, else cosine similarity.
        type          ``"text"`` | ``"table"`` | ``"image"``
        heading       Section heading above this chunk.
        pages         List of page indices (int).
        text          Full plain-text body — never truncated.
        table_html    Full HTML of the table (table chunks only; else ``""``)
        img_path      Relative image path (image chunks only; else ``""``)
        caption       Caption string (table / image chunks).
        ============  ===================================================
    """
    k = top_k if top_k is not None else settings.default_top_k
    fetch_k = max(
        candidate_k if candidate_k is not None else settings.retrieval_candidate_k,
        k,
    )

    # ── Stage 1: bi-encoder search ────────────────────────────────────────────
    q_emb = model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=fetch_k,
        include=["metadatas", "distances", "documents"],
    )

    # ── Stage 2: hydrate assets + build intermediate result list ──────────────
    candidates: list[ResultChunk] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        source_idx = meta.get("source_idx", -1)
        table_html = ""
        img_path = meta.get("img_path", "")

        if 0 <= source_idx < len(content):
            raw = content[source_idx]
            if meta["type"] == "table":
                table_html = raw.get("table_body", "")
            elif meta["type"] == "image":
                img_path = raw.get("img_path", img_path)

        candidates.append(
            {
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

    # ── Stage 3: cross-encoder reranking ─────────────────────────────────────
    if reranker is not None:
        candidates = _rerank(query, candidates, reranker)

    # Assign final ranks and truncate
    for rank, chunk in enumerate(candidates[:k], start=1):
        chunk["rank"] = rank

    return candidates[:k]


def _rerank(
    query: str,
    candidates: list[ResultChunk],
    reranker: CrossEncoder,
) -> list[ResultChunk]:
    """Score each (query, chunk_text) pair with the cross-encoder and re-sort.

    The text sent to the reranker is the same text used at embed time
    (heading + body for text, heading + caption for tables/images). This
    keeps scoring consistent with what was indexed.

    Args:
        query:      The user query string.
        candidates: Hydrated result dicts from Stage 2.
        reranker:   Loaded CrossEncoder model.

    Returns:
        ``candidates`` sorted descending by cross-encoder score, with the
        ``score`` field overwritten by the reranker logit.
    """
    # Build (query, passage) pairs — prefer full text, fall back to caption
    pairs = [(query, _rerank_text(c)) for c in candidates]

    scores: list[float] = reranker.predict(pairs).tolist()

    for chunk, score in zip(candidates, scores):
        chunk["score"] = round(score, 4)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _rerank_text(chunk: ResultChunk) -> str:
    """Select the most informative text representation of a chunk for reranking."""
    if chunk["type"] == "text":
        passage = f"{chunk['heading']}\n{chunk['text']}"
    elif chunk["type"] == "table":
        # Use plain-text table body; fall back to caption if empty
        passage = chunk["text"] or chunk["caption"]
    else:  # image
        passage = chunk["caption"] or "(image)"

    return passage.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Context serialiser
# ─────────────────────────────────────────────────────────────────────────────


def format_chunks_as_context(chunks: list[ResultChunk]) -> str:
    """Serialise retrieved chunks into a labelled context block for an LLM."""
    parts: list[str] = []
    for chunk in chunks:
        header = (
            f"[{chunk['type'].upper()} | heading: {chunk['heading']} "
            f"| pages: {chunk['pages']}]"
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
_singleton_reranker: CrossEncoder | None = None


def get_retrieval_resources() -> (
    tuple[chromadb.Collection, SentenceTransformer, ContentList, CrossEncoder]
):
    """Return ``(collection, model, content_list, reranker)``, initialising lazily."""
    global _singleton_collection, _singleton_model, _singleton_content, _singleton_reranker

    if _singleton_model is None:
        print(f"[info] Loading embedding model {settings.embedding_model} …")
        _singleton_model = SentenceTransformer(settings.embedding_model)

    if _singleton_reranker is None:
        print(f"[info] Loading reranker model {settings.reranker_model} …")
        _singleton_reranker = CrossEncoder(settings.reranker_model)

    if _singleton_collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        _singleton_collection = client.get_collection(settings.collection_name)
        print(f"[info] Connected to ChromaDB collection '{settings.collection_name}'")

    if _singleton_content is None and settings.content_list_path.exists():
        with open(settings.content_list_path, encoding="utf-8") as fh:
            _singleton_content = json.load(fh)

    return (
        _singleton_collection,
        _singleton_model,
        _singleton_content or [],
        _singleton_reranker,
    )


def query_documents(query: str, top_k: int | None = None) -> list[ResultChunk]:
    """Embed *query*, rerank candidates, and return the top-k most relevant chunks.

    Args:
        query:  Natural-language question.
        top_k:  Number of results. Defaults to ``settings.default_top_k``.

    Returns:
        List of result dicts as described in ``retrieve()``.
    """
    collection, model, content, reranker = get_retrieval_resources()
    return retrieve(query, collection, model, content, top_k=top_k, reranker=reranker)
