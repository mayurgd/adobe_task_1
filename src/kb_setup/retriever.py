"""
retriever.py

Semantic search + reranking against a specific document's ChromaDB collection.

Pipeline:
    1. Bi-encoder embeds the query → ChromaDB returns top candidates
    2. Cross-encoder reranks → returns top_k results

Public API:
    query_documents(query, collection_name, top_k) -> list[dict]
    format_chunks_as_context(chunks)               -> str
"""

from __future__ import annotations

import json

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Model singletons (expensive to load, shared across calls) ─────────────────
_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def _get_models() -> tuple[SentenceTransformer, CrossEncoder]:
    global _model, _reranker
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _model, _reranker


def _get_collection(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    return client.get_collection(collection_name)


# ─────────────────────────────────────────────────────────────────────────────


def query_documents(
    query: str,
    collection_name: str,
    top_k: int | None = None,
) -> list[dict]:
    """Query a single document's collection.

    Args:
        query:           Natural-language question.
        collection_name: Collection name from doc_registry.collection_name_for().
        top_k:           Chunks to return (default: settings.default_top_k).
    """
    k = top_k or settings.default_top_k
    fetch_k = max(settings.retrieval_candidate_k, k)
    model, reranker = _get_models()
    collection = _get_collection(collection_name)

    # Stage 1 — semantic search
    q_emb = model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=fetch_k,
        include=["metadatas", "distances"],
    )

    # Stage 2 — hydrate
    candidates = []
    for i, (meta, dist) in enumerate(
        zip(results["metadatas"][0], results["distances"][0])
    ):
        candidates.append(
            {
                "score": round(1 - dist, 4),
                "heading": meta["heading"],
                "text": meta["text"],
                "pages": json.loads(meta["pages"]),
                "tables": json.loads(meta.get("tables", "[]")),
                "images": json.loads(meta.get("images", "[]")),
            }
        )

    # Stage 3 — rerank
    pairs = [(query, _rerank_text(c)) for c in candidates]
    scores = reranker.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["score"] = round(s, 4)
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Drop negatives (cross-encoder: negative = not relevant)
    candidates = [c for c in candidates if c["score"] >= 0] or []

    for rank, c in enumerate(candidates[:k], start=1):
        c["rank"] = rank
    return candidates[:k]


def format_chunks_as_context(chunks: list[dict]) -> str:
    """Serialise chunks into a labelled context string for the LLM."""
    parts = []
    for chunk in chunks:
        header = f"[heading: {chunk['heading']} | pages: {chunk['pages']}]"
        body = []
        if chunk["text"]:
            body.append(chunk["text"])
        for i, t in enumerate(chunk.get("tables", []), 1):
            label = (
                f"[TABLE {i}" + (f" — {t['caption']}" if t.get("caption") else "") + "]"
            )
            body.append(f"{label}\n{t['text'] or t['html']}")
        for i, img in enumerate(chunk.get("images", []), 1):
            label = (
                f"[IMAGE {i}"
                + (f" — {img['caption']}" if img.get("caption") else "")
                + "]"
            )
            body.append(label)
        parts.append(f"{header}\n" + "\n\n".join(body))
    return "\n\n---\n\n".join(parts)


def _rerank_text(chunk: dict) -> str:
    parts = [
        (
            f"{chunk['heading']}\n{chunk['text']}".strip()
            if chunk["text"]
            else chunk["heading"]
        )
    ]
    for t in chunk.get("tables", []):
        lines = [l for l in t.get("text", "").splitlines() if l.strip()]
        col_headers = t.get("col_headers", "")
        # If pandas assigned auto-integer headers (e.g. "0 | 1 | 2"), skip them
        # and treat the first plain-text line as the header instead
        if col_headers and all(p.strip().isdigit() for p in col_headers.split("|")):
            col_headers = lines[0] if lines else ""
            data_lines = lines[1:3]
        else:
            data_lines = lines[1:3]
        digest = "\n".join(
            p for p in [t.get("caption", ""), col_headers, "\n".join(data_lines)] if p
        )
        parts.append(digest.strip())
    return "\n\n".join(p for p in parts if p)
