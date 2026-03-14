"""
indexer.py

Embeds a list of chunks (produced by chunker.py) and persists them in a
ChromaDB collection.

Full chunk text is stored directly in ChromaDB metadata — no truncation,
no sidecar file. This mirrors how table text has always been stored.
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

Chunk = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and return the SentenceTransformer embedding model."""
    name = model_name or settings.embedding_model
    print(f"[info] Loading embedding model: {name} …")
    return SentenceTransformer(name)


# ─────────────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────────────


def build_index(
    chunks: list[Chunk],
    model: SentenceTransformer | None = None,
    *,
    chroma_db_path: str | None = None,
    collection_name: str | None = None,
) -> chromadb.Collection:
    """Embed *chunks* and store them in a (re-created) ChromaDB collection.

    Full chunk text is stored in metadata with no truncation — identical to
    how table text is handled.

    Args:
        chunks:           List of chunk dicts as returned by ``build_chunks()``.
        model:            Optional pre-loaded SentenceTransformer.
        chroma_db_path:   Override for ``settings.chroma_db_path``.
        collection_name:  Override for ``settings.collection_name``.

    Returns:
        The populated ``chromadb.Collection`` instance.
    """
    if model is None:
        model = load_embedding_model()

    db_path = chroma_db_path or settings.chroma_db_path
    coll_name = collection_name or settings.collection_name

    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[indexer] Embedding {len(chunks)} chunks with {settings.embedding_model} …")
    embed_texts = [c["embed_text"] for c in chunks]
    embeddings = model.encode(
        embed_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=settings.embed_batch_size,
    ).tolist()

    ids = [c["id"] for c in chunks]
    metadatas = _build_metadatas(chunks)

    batch = settings.index_batch_size
    for i in range(0, len(chunks), batch):
        end = min(i + batch, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=embed_texts[i:end],
        )
        print(f"  Stored {end}/{len(chunks)} chunks")

    return collection


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_metadatas(chunks: list[Chunk]) -> list[dict]:
    """Serialise each chunk into a ChromaDB-compatible metadata dict.

    ``text`` is stored in full — no truncation. ChromaDB metadata values must
    be ``str | int | float | bool``; lists (``pages``) are JSON-serialised.
    """
    return [
        {
            "type": c["type"],
            "heading": c["heading"],
            "text": c["text"],
            "source_idx": c.get("source_idx", -1),
            "img_path": c.get("img_path", ""),
            "caption": c.get("caption", ""),
            "pages": json.dumps(c["pages"]),
        }
        for c in chunks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def log_chunk_stats(chunks: list[Chunk]) -> None:
    """Print a brief summary of chunk counts by type to stdout."""
    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"Total chunks : {len(chunks)}")
    for t, count in sorted(type_counts.items()):
        print(f"  {t:8s}: {count}")
