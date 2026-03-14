"""
index_documents.py

CLI entry point — builds the ChromaDB vector index from a MinerU
``_content_list.json`` file.

Pipeline
--------
1. Load the raw content_list JSON.
2. Build typed chunks           (chunker.build_chunks)
3. Embed + store in ChromaDB    (indexer.build_index)
4. Smoke-test retrieval         (retriever.retrieve)

Usage:
    python index_documents.py

The script is idempotent: re-running it drops and recreates the collection.

Note: ``query_documents`` and ``format_chunks_as_context`` are also re-exported
from this module so that any legacy import of the form
``from index_documents import query_documents`` keeps working.
"""

from __future__ import annotations

import json

from kb_setup.chunker import build_chunks
from config import settings
from kb_setup.indexer import build_index, load_embedding_model, log_chunk_stats

# Re-export for backward compatibility
from retriever import format_chunks_as_context, query_documents, retrieve


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── 1. Load content list ──────────────────────────────────────────────────
    print("=== Loading content list ===")
    with open(settings.content_list_path, encoding="utf-8") as f:
        content = json.load(f)
    print(f"Loaded {len(content)} raw elements")

    # ── 2. Build chunks ───────────────────────────────────────────────────────
    print("\n=== Building chunks ===")
    chunks = build_chunks(content)
    log_chunk_stats(chunks)

    # ── 3. Embed + index ──────────────────────────────────────────────────────
    print("\n=== Loading embedding model ===")
    model = load_embedding_model()

    print("\n=== Building ChromaDB index ===")
    collection = build_index(chunks, model)
    print(f"\nIndex saved to : {settings.chroma_db_path}")
    print(f"Collection     : {settings.collection_name}")
    print(f"Total vectors  : {collection.count()}")


if __name__ == "__main__":
    main()
