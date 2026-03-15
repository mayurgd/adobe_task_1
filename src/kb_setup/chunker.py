"""
chunker.py  (patched)

Only change from original:
  - _Section tracks `first_source_idx` in addition to `last_source_idx`
  - flush() emits both as `source_idx_start` and `source_idx` (end) on every chunk

Everything else is identical to the original.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import settings
from kb_setup.text_utils import clean, table_column_headers, table_html_to_text

ContentList = list[dict[str, Any]]
Chunk = dict[str, Any]

MAX_TOKENS = 480
OVERLAP_TOKENS = 50

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


class _Section:
    """Mutable accumulator for one heading's worth of content."""

    def __init__(self, heading: str, heading_source_idx: int = -1) -> None:
        self.heading = heading
        self.text_parts: list[str] = []
        self.pages: set[int] = set()
        # first_source_idx includes the heading item itself
        self.first_source_idx: int = heading_source_idx
        self.last_source_idx: int = heading_source_idx
        self.tables: list[dict] = []
        self.images: list[dict] = []

    def _record_idx(self, source_idx: int) -> None:
        if self.first_source_idx == -1:
            self.first_source_idx = source_idx
        self.last_source_idx = source_idx

    def add_text(self, text: str, page: int, source_idx: int) -> None:
        self.text_parts.append(text)
        self.pages.add(page)
        self._record_idx(source_idx)

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
        self._record_idx(source_idx)

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
        self._record_idx(source_idx)

    def is_empty(self) -> bool:
        body = "\n".join(self.text_parts)
        return (
            len(body) < settings.min_text_length and not self.tables and not self.images
        )

    def flush(self) -> list[Chunk]:
        if self.is_empty():
            return []

        body = "\n".join(self.text_parts)
        pages = sorted(self.pages)
        tables = self.tables
        images = self.images

        combined = f"{self.heading}\n{body}".strip()
        if _count_tokens(combined) <= MAX_TOKENS or not body.strip():
            splits = [body] if body.strip() else [""]
        else:
            splits = _get_splitter(self.heading).split_text(body)

        chunks: list[Chunk] = []
        for split_body in splits:
            split_body = split_body.strip()
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
                    # ── PATCHED: store the full range ─────────────────────────
                    "source_idx_start": self.first_source_idx,
                    "source_idx": self.last_source_idx,  # kept for compat
                    # ─────────────────────────────────────────────────────────
                    "tables": tables,
                    "images": images,
                    "img_path": images[0]["img_path"] if images else "",
                    "caption": "",
                }
            )

        return chunks


def build_chunks(content: ContentList) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = _Section(heading="", heading_source_idx=-1)

    for source_idx, item in enumerate(content):
        item_type = item.get("type")

        if item_type == "text":
            text = clean(item.get("text", ""))
            if not text:
                continue
            if item.get("text_level"):
                # Flush previous section, start new one with heading's own index
                chunks.extend(section.flush())
                section = _Section(heading=text, heading_source_idx=source_idx)
            else:
                section.add_text(text, item.get("page_idx", 0), source_idx)

        elif item_type == "table":
            section.add_table(item, source_idx)

        elif item_type == "image":
            section.add_image(item, source_idx)

    chunks.extend(section.flush())
    return chunks
