"""
chunker.py

Converts a flat MinerU ``_content_list.json`` into typed chunks ready for
embedding and indexing.

Two-stage retrieval design
--------------------------
Stage 1 (embedding side, this module):
    Each chunk stores a compact ``embed_text`` — the string that gets embedded:
        • text   : heading + paragraph body
        • table  : heading + caption + column headers (NOT the full cell data)
        • image  : heading + caption

Stage 2 (query-time, handled by retriever.py):
    Full assets are fetched from the original content_list via ``source_idx``
    so that bulky HTML is never stored in ChromaDB metadata.

Public API:
    build_chunks(content: list) -> list[dict]
"""

from __future__ import annotations

import uuid
from typing import Any

from config import settings
from kb_setup.text_utils import clean, table_column_headers, table_html_to_text

# ── Type alias ────────────────────────────────────────────────────────────────
ContentList = list[dict[str, Any]]
Chunk = dict[str, Any]


def _make_text_chunk(
    heading: str,
    text_parts: list[str],
    pages: set[int],
) -> Chunk | None:
    """Assemble and return a text chunk, or None if the body is too short."""
    if not text_parts:
        return None
    body = "\n".join(text_parts)
    if len(body) < settings.min_text_length:
        return None
    return {
        "id": str(uuid.uuid4()),
        "embed_text": clean(f"{heading}\n{body}"),
        "type": "text",
        "heading": heading,
        "text": body,
        "pages": sorted(pages),
        "source_idx": -1,
        "img_path": "",
        "caption": "",
    }


def _make_table_chunk(
    heading: str,
    item: dict[str, Any],
    source_idx: int,
) -> Chunk:
    """Build a table chunk from a MinerU table element."""
    html = item.get("table_body", "")
    caption_list = item.get("table_caption") or []
    caption = clean(caption_list[0]) if caption_list else ""

    col_headers = table_column_headers(html)
    embed_text = clean(f"{heading}\n{caption}\n{col_headers}")

    return {
        "id": str(uuid.uuid4()),
        "embed_text": embed_text,
        "type": "table",
        "heading": heading,
        "text": table_html_to_text(html),
        "source_idx": source_idx,
        "img_path": "",
        "caption": caption,
        "pages": [item.get("page_idx", 0)],
    }


def _make_image_chunk(
    heading: str,
    item: dict[str, Any],
    source_idx: int,
) -> Chunk:
    """Build an image chunk from a MinerU image element."""
    caption_list = item.get("image_caption") or []
    caption = clean(caption_list[0]) if caption_list else ""
    img_path = item.get("img_path", "")

    return {
        "id": str(uuid.uuid4()),
        "embed_text": clean(f"{heading}\n{caption}"),
        "type": "image",
        "heading": heading,
        "text": caption,
        "source_idx": source_idx,
        "img_path": img_path,
        "caption": caption,
        "pages": [item.get("page_idx", 0)],
    }


def build_chunks(content: ContentList) -> list[Chunk]:
    """Walk the flat content list in reading order and produce typed chunks.

    Each returned chunk dict contains:

    ============  ============================================================
    Key           Description
    ============  ============================================================
    id            UUID string — unique per chunk.
    embed_text    Text that will be embedded (compact; excludes full cell data).
    type          ``"text"`` | ``"table"`` | ``"image"``
    heading       Nearest heading above this chunk in reading order.
    pages         Sorted list of page indices (int).
    text          Plain-text body (paragraphs, stringified table, or caption).
    source_idx    Index into the original content_list (–1 for text chunks).
                  Used at query-time to hydrate full HTML / image metadata.
    img_path      Relative image path (image chunks; convenience copy).
    caption       Caption string (table / image chunks).
    ============  ============================================================

    Args:
        content: Parsed ``_content_list.json`` as a list of dicts.

    Returns:
        List of chunk dicts in document reading order.
    """
    chunks: list[Chunk] = []

    current_heading = ""
    current_text_parts: list[str] = []
    current_pages: set[int] = set()

    def _flush() -> None:
        """Flush pending text parts into a chunk (if substantial enough)."""
        nonlocal current_text_parts, current_pages
        chunk = _make_text_chunk(current_heading, current_text_parts, current_pages)
        if chunk:
            chunks.append(chunk)
        current_text_parts = []
        current_pages = set()

    for source_idx, item in enumerate(content):
        item_type = item.get("type")

        if item_type == "text":
            text = clean(item.get("text", ""))
            if not text:
                continue
            if item.get("text_level"):  # heading — start new section
                _flush()
                current_heading = text
            else:  # paragraph — accumulate
                current_text_parts.append(text)
                current_pages.add(item.get("page_idx", 0))

        elif item_type == "table":
            _flush()
            chunks.append(_make_table_chunk(current_heading, item, source_idx))

        elif item_type == "image":
            # Images don't interrupt paragraph flow — no flush needed
            chunks.append(_make_image_chunk(current_heading, item, source_idx))

        # Unknown / unsupported types are silently ignored

    _flush()  # flush final section

    return chunks
