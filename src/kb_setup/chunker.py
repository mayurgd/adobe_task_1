"""
chunker.py

Converts a flat MinerU ``_content_list.json`` into typed chunks ready for
embedding and indexing.

Chunking strategy
-----------------
Heading-based grouping is the primary split. Everything under a heading —
paragraphs AND tables — is accumulated together and emitted as a **single
chunk** when the next heading (or end-of-document) is reached.

This preserves section semantics: for example, "Financial Targets" stays as
one chunk even if it contains multiple paragraphs and multiple tables.

Chunk schema
------------
Each chunk has:
  - embed_text : heading + all paragraph text (used for vector search)
  - text       : same as embed_text body (for reranker / context display)
  - tables     : list of table dicts  {caption, text, html, source_idx, page}
  - images     : list of image dicts  {caption, img_path, source_idx, page}

Tables and images are stored as structured metadata on the parent chunk —
they are NOT emitted as separate chunks. This keeps the full section together
for retrieval.

If the accumulated text body exceeds MAX_TOKENS, it is split via LangChain's
RecursiveCharacterTextSplitter (token-aware). Each split inherits the full
tables/images list so that every sub-chunk stays associated with its section's
assets.

Public API:
    build_chunks(content: list) -> list[dict]
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import settings
from kb_setup.text_utils import clean, table_column_headers, table_html_to_text

# ── Type aliases ──────────────────────────────────────────────────────────────
ContentList = list[dict[str, Any]]
Chunk = dict[str, Any]

# ── Token budget ──────────────────────────────────────────────────────────────
MAX_TOKENS = 480
OVERLAP_TOKENS = 50

# ── Lazy singletons ───────────────────────────────────────────────────────────
_tokenizer: AutoTokenizer | None = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _get_splitter(heading: str) -> RecursiveCharacterTextSplitter:
    heading_tokens = _count_tokens(heading)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_tokenizer(),
        chunk_size=max(64, MAX_TOKENS - heading_tokens),
        chunk_overlap=OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section state — accumulates everything under one heading
# ─────────────────────────────────────────────────────────────────────────────


class _Section:
    """Mutable accumulator for one heading's worth of content."""

    def __init__(self, heading: str) -> None:
        self.heading = heading
        self.text_parts: list[str] = []
        self.pages: set[int] = set()
        self.last_source_idx: int = -1
        # Tables and images stay as structured metadata on the chunk
        self.tables: list[dict] = []
        self.images: list[dict] = []

    def add_text(self, text: str, page: int, source_idx: int) -> None:
        self.text_parts.append(text)
        self.pages.add(page)
        self.last_source_idx = source_idx

    def add_table(self, item: dict, source_idx: int) -> None:
        html = item.get("table_body", "")
        caption_list = item.get("table_caption") or []
        caption = clean(caption_list[0]) if caption_list else ""
        self.tables.append(
            {
                "caption": caption,
                "text": table_html_to_text(html),
                "html": html,
                "col_headers": table_column_headers(html),
                "source_idx": source_idx,
                "page": item.get("page_idx", 0),
            }
        )
        self.pages.add(item.get("page_idx", 0))
        self.last_source_idx = source_idx

    def add_image(self, item: dict, source_idx: int) -> None:
        caption_list = item.get("image_caption") or []
        caption = clean(caption_list[0]) if caption_list else ""
        self.images.append(
            {
                "caption": caption,
                "img_path": item.get("img_path", ""),
                "source_idx": source_idx,
                "page": item.get("page_idx", 0),
            }
        )
        self.pages.add(item.get("page_idx", 0))

    def is_empty(self) -> bool:
        body = "\n".join(self.text_parts)
        return (
            len(body) < settings.min_text_length and not self.tables and not self.images
        )

    def flush(self) -> list[Chunk]:
        """Emit one (or more, if over token limit) chunks for this section."""
        if self.is_empty():
            return []

        body = "\n".join(self.text_parts)
        pages = sorted(self.pages)
        tables = self.tables
        images = self.images

        # embed_text = heading + paragraph text only (tables too noisy for bi-encoder)
        combined = f"{self.heading}\n{body}".strip()
        if _count_tokens(combined) <= MAX_TOKENS or not body.strip():
            splits = [body] if body.strip() else [""]
        else:
            splits = _get_splitter(self.heading).split_text(body)

        chunks: list[Chunk] = []
        for split_body in splits:
            split_body = split_body.strip()
            # Always emit if section has tables/images, even if text is short
            if len(split_body) < settings.min_text_length and not tables and not images:
                continue

            embed_text = (
                clean(f"{self.heading}\n{split_body}")
                if split_body
                else clean(self.heading)
            )

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "embed_text": embed_text,
                    "type": "text",
                    "heading": self.heading,
                    "text": split_body,
                    "pages": pages,
                    "source_idx": self.last_source_idx,
                    # Structured assets — stored as JSON-serialisable lists
                    "tables": tables,
                    "images": images,
                    # Legacy flat fields kept for backward compat with retriever
                    "img_path": images[0]["img_path"] if images else "",
                    "caption": "",
                }
            )

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────


def build_chunks(content: ContentList) -> list[Chunk]:
    """Walk the flat content list in reading order and produce typed chunks.

    Everything under a heading (paragraphs, tables, images) is collected into
    a single chunk. Tables and images are embedded in the chunk's ``tables``
    and ``images`` metadata lists rather than being emitted as separate chunks.

    Each returned chunk dict contains:

    ============  ============================================================
    Key           Description
    ============  ============================================================
    id            UUID string — unique per chunk.
    embed_text    Text sent to the bi-encoder. <= MAX_TOKENS tokens.
                  = heading + paragraph text (tables omitted — too noisy).
    type          Always ``"text"`` (tables/images are metadata, not chunks).
    heading       Section heading.
    text          Full paragraph body for this chunk (may be a split).
    pages         Sorted list of all page indices in this section.
    source_idx    Index of the last contributing item in the content_list.
    tables        List of table dicts: {caption, text, html, col_headers,
                  source_idx, page}. Empty list if section has no tables.
    images        List of image dicts: {caption, img_path, source_idx, page}.
                  Empty list if section has no images.
    img_path      First image path for backward compat ('' if none).
    caption       Empty string (captions live inside tables/images lists).
    ============  ============================================================

    Args:
        content: Parsed ``_content_list.json`` as a list of dicts.

    Returns:
        List of chunk dicts in document reading order.
    """
    chunks: list[Chunk] = []
    section = _Section(heading="")

    for source_idx, item in enumerate(content):
        item_type = item.get("type")

        if item_type == "text":
            text = clean(item.get("text", ""))
            if not text:
                continue

            if item.get("text_level"):
                # New heading → flush current section, start a new one
                chunks.extend(section.flush())
                section = _Section(heading=text)
            else:
                # Paragraph → accumulate into current section
                section.add_text(text, item.get("page_idx", 0), source_idx)

        elif item_type == "table":
            # Table → attach to current section as metadata (no separate chunk)
            section.add_table(item, source_idx)

        elif item_type == "image":
            # Image → attach to current section as metadata (no separate chunk)
            section.add_image(item, source_idx)

    # Flush the final section
    chunks.extend(section.flush())

    return chunks
