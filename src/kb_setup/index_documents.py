"""
index_documents.py

Indexes a document into its own ChromaDB collection and registers it.
Called by the UI upload handler or directly as a CLI.

Usage:
    python index_documents.py --file "Q2 Results.pdf" --content-list /path/to/_content_list.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from config import settings
from doc_registry import collection_name_for, register
from kb_setup.chunker import build_chunks
from kb_setup.indexer import build_index, load_embedding_model


def index_file(filename: str, content_list_path: str) -> dict:
    """Index a single document. Re-indexing the same filename overwrites.

    Args:
        filename:          Original filename e.g. "Q2 Results.pdf"
        content_list_path: Path to the MinerU _content_list.json

    Returns:
        {"filename", "collection_name", "chunk_count"}
    """
    filename = Path(filename).name  # always use basename, never full path
    coll_name = collection_name_for(filename)

    with open(content_list_path, encoding="utf-8") as f:
        content = json.load(f)

    chunks = build_chunks(content)
    model = load_embedding_model()
    build_index(chunks, model, collection_name=coll_name)
    register(filename, coll_name, len(chunks))

    print(f"Indexed '{filename}' → collection '{coll_name}' ({len(chunks)} chunks)")
    return {
        "filename": filename,
        "collection_name": coll_name,
        "chunk_count": len(chunks),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Original filename")
    parser.add_argument(
        "--content-list", required=True, help="Path to _content_list.json"
    )
    args = parser.parse_args()
    index_file(args.file, args.content_list)
