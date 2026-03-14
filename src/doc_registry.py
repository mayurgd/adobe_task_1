"""
doc_registry.py

Maps uploaded documents to their ChromaDB collection names.
Stored as data/doc_registry.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "doc_registry.json"


def collection_name_for(filename: str) -> str:
    """Derive a stable ChromaDB collection name from a filename.

    e.g. "Q2 Results.pdf" -> "doc_q2_results_a3f1b2c4"
    """
    stem = Path(filename).name  # strip any leading path
    stem = Path(stem).stem
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:40]
    short_hash = hashlib.sha1(filename.encode()).hexdigest()[:8]
    return f"doc_{slug}_{short_hash}"


def load() -> dict:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text())
    return {}


def save(data: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def register(filename: str, collection_name: str, chunk_count: int) -> None:
    data = load()
    data[collection_name] = {
        "filename": filename,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "indexed_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    save(data)


def list_all() -> list[dict]:
    return sorted(load().values(), key=lambda e: e["indexed_at"], reverse=True)


def remove(collection_name: str) -> None:
    data = load()
    data.pop(collection_name, None)
    save(data)
