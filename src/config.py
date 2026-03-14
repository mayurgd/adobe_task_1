"""
config.py

Single source of truth for all environment variables, file paths, and
model/index constants used across the project.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # ── Langfuse (optional) ───────────────────────────────────────────────────
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── ChromaDB / embedding ──────────────────────────────────────────────────
    collection_name: str = "leadership_insight_docs"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ── MinerU ────────────────────────────────────────────────────────────────
    mineru_output_dir: str = str(_REPO_ROOT / "data" / "docs" / "outputs")

    # ── Chunking ──────────────────────────────────────────────────────────────
    min_text_length: int = 30

    # ── Retrieval / indexing ──────────────────────────────────────────────────
    default_top_k: int = 5
    index_batch_size: int = 500
    embed_batch_size: int = 64

    # ── Derived fields ────────────────────────────────────────────────────────
    @computed_field
    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @computed_field
    @property
    def chroma_db_path(self) -> str:
        return str(_REPO_ROOT / "data" / "chroma_db")

    @computed_field
    @property
    def content_list_path(self) -> Path:
        """Return the most recently created _content_list.json under mineru_output_dir."""
        base = Path(self.mineru_output_dir)
        matches = list(base.glob("*/auto/*_content_list.json"))
        if not matches:
            return base
        return max(matches, key=lambda p: p.stat().st_mtime)

    @computed_field
    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # ── Startup validation ────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _warn_if_tracing_incomplete(self) -> "Settings":
        keys = (self.langfuse_public_key, self.langfuse_secret_key)
        if any(keys) and not all(keys):
            import warnings

            warnings.warn(
                "Only one Langfuse key is set — tracing will be disabled. "
                "Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable it.",
                stacklevel=2,
            )
        return self


settings = Settings()
