"""
retriever.py

Query-time retrieval: semantic search over the ChromaDB collection, two-stage
asset hydration, cross-encoder reranking, and context serialisation for LLM
consumption.

Retrieval pipeline
------------------
Stage 1 — Semantic search (bi-encoder)
    The query is embedded with BGE and ChromaDB returns the top
    ``retrieval_candidate_k`` candidates by cosine similarity.

Stage 2 — Asset hydration
    ``tables`` and ``images`` are deserialised from their JSON-string metadata.
    Each chunk now carries its full section context — paragraphs AND tables.

Stage 3 — Cross-encoder reranking
    A cross-encoder scores every (query, chunk_text) pair and re-sorts.

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
# Core retrieval
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

    Each returned result dict contains:

    ============  ===================================================
    Key           Description
    ============  ===================================================
    rank          1-based rank (int).
    score         Reranker logit if reranked, else cosine similarity.
    type          Always ``"text"`` (tables are metadata, not chunks).
    heading       Section heading.
    pages         List of page indices (int).
    text          Full paragraph body for this chunk.
    tables        List of table dicts: {caption, text, html,
                  col_headers, source_idx, page}.
    images        List of image dicts: {caption, img_path,
                  source_idx, page}.
    img_path      First image path ('' if none) — backward compat.
    caption       '' — captions live inside tables/images lists.
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

    # ── Stage 2: hydrate assets ───────────────────────────────────────────────
    candidates: list[ResultChunk] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Deserialise JSON-string lists back to Python objects
        tables = json.loads(meta.get("tables", "[]"))
        images = json.loads(meta.get("images", "[]"))

        candidates.append(
            {
                "score": round(1 - distance, 4),
                "type": meta["type"],
                "heading": meta["heading"],
                "pages": json.loads(meta["pages"]),
                "text": meta["text"],
                "tables": tables,
                "images": images,
            }
        )

    # ── Stage 3: cross-encoder reranking ─────────────────────────────────────
    if reranker is not None:
        candidates = _rerank(query, candidates, reranker)

    for rank, chunk in enumerate(candidates[:k], start=1):
        chunk["rank"] = rank

    return candidates[:k]


def _rerank(
    query: str,
    candidates: list[ResultChunk],
    reranker: CrossEncoder,
) -> list[ResultChunk]:
    pairs = [(query, _rerank_text(c)) for c in candidates]
    scores: list[float] = reranker.predict(pairs).tolist()
    for chunk, score in zip(candidates, scores):
        chunk["score"] = round(score, 4)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Drop chunks the cross-encoder considers irrelevant (negative logit).
    # ms-marco cross-encoders are not calibrated — negative scores reliably
    # indicate the passage is not relevant to the query.
    positives = [c for c in candidates if c["score"] >= 0]
    # Guard: if everything is negative (very off-topic query), return the
    # single best candidate rather than an empty list.
    return positives if positives else candidates[:1]


def _rerank_text(chunk: ResultChunk) -> str:
    """Build the passage text sent to the cross-encoder.

    Paragraph text is used as-is. For tables we send a compact digest:
        caption (if any) + column headers + first 2 data rows.

    This gives the reranker enough signal to judge relevance without the
    noise and token cost of full table dumps.
    """
    parts = []
    if chunk["text"]:
        parts.append(f"{chunk['heading']}\n{chunk['text']}")
    else:
        parts.append(chunk["heading"])

    for t in chunk.get("tables", []):
        table_parts = []
        if t.get("caption"):
            table_parts.append(t["caption"])
        table_parts.append(_table_digest(t))
        parts.append("\n".join(table_parts))

    return "\n\n".join(parts).strip()


def _table_digest(table: dict, max_rows: int = 2) -> str:
    """Return column headers + up to *max_rows* data rows as a compact string.

    If pandas assigned auto-integer column indices (e.g. "0 | 1 | 2") because
    the table has no <th> header row, the stored col_headers are meaningless.
    In that case we treat the first data line from the plain-text as the header
    and take the next max_rows lines as data.
    """
    col_headers: str = table.get("col_headers", "")

    # Detect auto-integer headers like "0 | 1 | 2" — not useful for scoring
    if _is_auto_int_headers(col_headers):
        col_headers = ""

    plain: str = table.get("text", "")
    lines = [l for l in plain.splitlines() if l.strip()] if plain else []

    if col_headers:
        # Real headers available — skip the pandas header line, take max_rows data lines
        data_lines = lines[1 : max_rows + 1]
    else:
        # No real headers — first line IS the header, next max_rows are data
        col_headers = lines[0] if lines else ""
        data_lines = lines[1 : max_rows + 1]

    parts = [p for p in [col_headers, "\n".join(data_lines)] if p]
    return "\n".join(parts)


def _is_auto_int_headers(col_headers: str) -> bool:
    """Return True if col_headers looks like pandas auto-integer indices, e.g. '0 | 1 | 2'."""
    if not col_headers:
        return False
    return all(part.strip().isdigit() for part in col_headers.split("|"))


# ─────────────────────────────────────────────────────────────────────────────
# Context serialiser
# ─────────────────────────────────────────────────────────────────────────────


def format_chunks_as_context(chunks: list[ResultChunk]) -> str:
    """Serialise retrieved chunks into a labelled context block for an LLM.

    For each chunk, paragraph text is followed by any tables (as plain text)
    and image captions — keeping the full section together.
    """
    parts: list[str] = []
    for chunk in chunks:
        header = f"[TEXT | heading: {chunk['heading']} | pages: {chunk['pages']}]"
        body_parts = []
        if chunk["text"]:
            body_parts.append(chunk["text"])

        for idx, t in enumerate(chunk.get("tables", []), start=1):
            table_header = f"[TABLE {idx}"
            if t.get("caption"):
                table_header += f" — {t['caption']}"
            table_header += "]"
            body_parts.append(f"{table_header}\n{t['text'] or t['html']}")

        for idx, img in enumerate(chunk.get("images", []), start=1):
            img_label = f"[IMAGE {idx}"
            if img.get("caption"):
                img_label += f" — {img['caption']}"
            img_label += "]"
            body_parts.append(img_label)

        parts.append(f"{header}\n" + "\n\n".join(body_parts))

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ─────────────────────────────────────────────────────────────────────────────

_singleton_collection: chromadb.Collection | None = None
_singleton_model: SentenceTransformer | None = None
_singleton_content: ContentList | None = None
_singleton_reranker: CrossEncoder | None = None


def get_retrieval_resources() -> (
    tuple[chromadb.Collection, SentenceTransformer, ContentList, CrossEncoder]
):
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
    collection, model, content, reranker = get_retrieval_resources()
    return retrieve(query, collection, model, content, top_k=top_k, reranker=reranker)
