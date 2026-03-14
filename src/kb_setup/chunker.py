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
    Full text is hydrated from metadata for text chunks, and from the original
    content_list via ``source_idx`` for tables and images.

Chunking strategy
-----------------
Heading-based grouping is the primary split. If the resulting body exceeds
MAX_TOKENS when combined with its heading, it is passed to LangChain's
RecursiveCharacterTextSplitter (token-aware, using the same tokenizer as the
embedding model). The splitter tries separators in order:

    \n\n  ->  \n  ->  ". "  ->  " "  ->  ""

and merges pieces greedily up to (MAX_TOKENS - heading_tokens) with a
OVERLAP_TOKENS sliding window. This guarantees no embed_text is silently
truncated by the model.

Public API:
    build_chunks(content: list) -> list[dict]
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import settings
from kb_setup.text_utils import clean, table_column_headers, table_html_to_text

# ── Type alias ────────────────────────────────────────────────────────────────
ContentList = list[dict[str, Any]]
Chunk = dict[str, Any]

# ── Token budget ──────────────────────────────────────────────────────────────
MAX_TOKENS = 480  # safe margin under bge-base-en-v1.5's 512 hard limit
OVERLAP_TOKENS = 50  # token overlap carried into each successive sub-chunk

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
    """Return a splitter whose chunk_size budget is reduced by the heading cost.

    This ensures that ``heading + split_body`` always fits within MAX_TOKENS
    after the split, since LangChain only sees the body text.
    """
    heading_tokens = _count_tokens(heading)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_tokenizer(),
        chunk_size=max(64, MAX_TOKENS - heading_tokens),
        chunk_overlap=OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunk factories
# ─────────────────────────────────────────────────────────────────────────────


def _make_text_chunk(
    heading: str,
    text_parts: list[str],
    pages: set[int],
    source_idx: int,
) -> list[Chunk]:
    """Assemble text chunk(s) from *text_parts*, splitting if over token limit.

    Returns a list — one logical section can become multiple chunks when it
    exceeds MAX_TOKENS.
    """
    if not text_parts:
        return []

    body = "\n".join(text_parts)
    if len(body) < settings.min_text_length:
        return []

    # Split only if the combined heading + body exceeds the token limit
    combined_tokens = _count_tokens(f"{heading}\n{body}")
    if combined_tokens <= MAX_TOKENS:
        splits = [body]
    else:
        splits = _get_splitter(heading).split_text(body)

    chunks = []
    for split_body in splits:
        split_body = split_body.strip()
        if len(split_body) < settings.min_text_length:
            continue
        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "embed_text": clean(f"{heading}\n{split_body}"),
                "type": "text",
                "heading": heading,
                "text": split_body,
                "pages": sorted(pages),
                "source_idx": source_idx,
                "img_path": "",
                "caption": "",
            }
        )
    return chunks


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


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────


def build_chunks(content: ContentList) -> list[Chunk]:
    """Walk the flat content list in reading order and produce typed chunks.

    Each returned chunk dict contains:

    ============  ============================================================
    Key           Description
    ============  ============================================================
    id            UUID string - unique per chunk.
    embed_text    Text that will be embedded. Always <= MAX_TOKENS tokens.
    type          "text" | "table" | "image"
    heading       Nearest heading above this chunk in reading order.
    pages         Sorted list of page indices (int).
    text          Full plain-text body - never truncated.
    source_idx    Index of the last contributing item in the original
                  content_list. Used at query-time to hydrate full HTML /
                  image metadata for table and image chunks.
    img_path      Relative image path (image chunks; convenience copy).
    caption       Caption string (table / image chunks).
    ============  ============================================================

    Text chunks exceeding MAX_TOKENS are split via LangChain's
    RecursiveCharacterTextSplitter (token-aware) and re-merged with a
    OVERLAP_TOKENS overlap window. Table and image embed_texts are compact
    by design and are not split.

    Args:
        content: Parsed ``_content_list.json`` as a list of dicts.

    Returns:
        List of chunk dicts in document reading order.
    """
    chunks: list[Chunk] = []

    current_heading = ""
    current_text_parts: list[str] = []
    current_pages: set[int] = set()
    current_source_idx: int = -1

    def _flush() -> None:
        nonlocal current_text_parts, current_pages
        for chunk in _make_text_chunk(
            current_heading,
            current_text_parts,
            current_pages,
            current_source_idx,
        ):
            chunks.append(chunk)
        current_text_parts = []
        current_pages = set()

    for source_idx, item in enumerate(content):
        item_type = item.get("type")

        if item_type == "text":
            text = clean(item.get("text", ""))
            if not text:
                continue
            if item.get("text_level"):  # heading - flush and start new section
                _flush()
                current_heading = text
            else:  # paragraph - accumulate
                current_text_parts.append(text)
                current_pages.add(item.get("page_idx", 0))
                current_source_idx = source_idx

        elif item_type == "table":
            _flush()
            chunks.append(_make_table_chunk(current_heading, item, source_idx))

        elif item_type == "image":
            chunks.append(_make_image_chunk(current_heading, item, source_idx))

    _flush()  # flush final section

    return chunks
