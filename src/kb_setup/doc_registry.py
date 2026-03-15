"""
doc_registry.py

Single registry for all indexed documents.
Reads and writes  data/registry.json  (the same file server.py uses).

Schema (registry.json):
    {
      "doc_registry": {
        "<doc_id>": {
          "name":            str,   # original filename
          "path":            str,   # absolute path on disk
          "pages":           int,
          "collection_name": str,
          "chunk_count":     int,
          "indexed_at":      str    # ISO-8601 UTC
        }
      },
      "job_status": {
        "<doc_id>": { "status": str, "message": str }
      }
    }

Public API (unchanged — drop-in replacement):
    collection_name_for(filename)                          -> str
    register(filename, collection_name, chunk_count)       -> None
    list_all()                                             -> list[dict]
    remove(collection_name)                                -> None
    load()                                                 -> dict   (raw doc_registry section)
    save(data)                                             -> None   (raw doc_registry section)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

# ── Single registry file ──────────────────────────────────────────────────────
_REGISTRY_PATH = Path(__file__).parent.parent.parent / "data" / "registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — read/write the whole registry.json
# ─────────────────────────────────────────────────────────────────────────────


def _read_full() -> dict:
    """Return the full registry dict, creating defaults if the file is absent."""
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"doc_registry": {}, "job_status": {}}


def _write_full(data: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────


def collection_name_for(filename: str) -> str:
    """Derive a stable ChromaDB collection name from a filename.

    e.g. "Q2 Results.pdf" -> "doc_q2_results_a3f1b2c4"
    """
    stem = Path(filename).name  # strip any leading path components
    stem = Path(stem).stem
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:40]
    short_hash = hashlib.sha1(filename.encode()).hexdigest()[:8]
    return f"doc_{slug}_{short_hash}"


# ── Compatibility shim: load() / save() operate on the doc_registry section ──


def load() -> dict:
    """Return the doc_registry section (collection_name → entry).

    Backfills any entry missing 'indexed_at' so callers can always sort on it.
    """
    full = _read_full()
    section: dict = full.get("doc_registry", {})

    # Backfill missing fields for entries written by older code
    for entry in section.values():
        entry.setdefault("indexed_at", "1970-01-01T00:00:00")
        entry.setdefault("chunk_count", 0)

    return section


def save(data: dict) -> None:
    """Write back the doc_registry section (collection_name → entry)."""
    full = _read_full()
    full["doc_registry"] = data
    _write_full(full)


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────


def register(filename: str, collection_name: str, chunk_count: int) -> None:
    """Upsert an indexed document entry into the registry.

    Looks for an existing doc_id whose collection_name matches so it can
    update the full entry in-place (preserving path, pages, etc.).
    Falls back to writing a collection_name-keyed entry for standalone CLI use.
    """
    full = _read_full()
    doc_reg: dict = full.setdefault("doc_registry", {})

    now_iso = datetime.utcnow().isoformat(timespec="seconds")

    # Try to find a matching entry by collection_name and update it
    for entry in doc_reg.values():
        if entry.get("collection_name") == collection_name:
            entry["chunk_count"] = chunk_count
            entry["indexed_at"] = now_iso
            entry.setdefault("name", filename)
            _write_full(full)
            return

    # Fallback: create a new entry keyed by collection_name (CLI / notebook path)
    doc_reg[collection_name] = {
        "name": filename,
        "path": "",
        "pages": 0,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "indexed_at": now_iso,
    }
    _write_full(full)


def list_all() -> list[dict]:
    """Return all doc_registry entries sorted by indexed_at descending.

    Each returned dict is guaranteed to have 'indexed_at' and 'chunk_count'.
    Shape matches the old doc_registry.json schema for backwards compatibility:
        { filename, collection_name, chunk_count, indexed_at, ... }
    """
    section = load()  # already backfills missing fields
    entries = []
    for entry in section.values():
        entries.append(
            {
                # Fields the old API guaranteed
                "filename": entry.get("name", ""),
                "collection_name": entry.get("collection_name", ""),
                "chunk_count": entry.get("chunk_count", 0),
                "indexed_at": entry.get("indexed_at", "1970-01-01T00:00:00"),
                # Extra fields available in the unified schema
                "path": entry.get("path", ""),
                "pages": entry.get("pages", 0),
            }
        )
    return sorted(entries, key=lambda e: e["indexed_at"], reverse=True)


def remove(collection_name: str) -> None:
    """Remove an entry by collection_name."""
    full = _read_full()
    doc_reg: dict = full.get("doc_registry", {})

    # Remove by collection_name key (CLI path)
    doc_reg.pop(collection_name, None)

    # Also remove any entry whose collection_name field matches (server path)
    to_delete = [
        k for k, v in doc_reg.items() if v.get("collection_name") == collection_name
    ]
    for k in to_delete:
        doc_reg.pop(k, None)

    _write_full(full)
