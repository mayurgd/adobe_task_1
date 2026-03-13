"""
config.py

Single source of truth for all environment variables, file paths, and
model/index constants used across the project.

Uses pydantic-settings for automatic env var loading, type validation,
and fail-fast errors on missing required fields.

Usage:
    from config import settings
    print(settings.openai_model)
    print(settings.chroma_db_path)

Install:
    pip install pydantic-settings
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")


class Settings(BaseSettings):
    """
    All configuration is read from environment variables or the .env file.
    Required fields (no default) raise a ValidationError at startup if missing.
    """

    model_config = SettingsConfigDict()

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str  # required — no default
    openai_model: str = "gpt-4o-mini"

    # ── Langfuse (optional) ───────────────────────────────────────────────────
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── ChromaDB / embedding ──────────────────────────────────────────────────
    collection_name: str = "adbe_2023_annual_report"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ── Chunking ──────────────────────────────────────────────────────────────
    min_text_length: int = 30  # minimum chars for a text chunk to be indexed
    metadata_text_cap: int = 2000  # max chars stored in ChromaDB metadata

    # ── Retrieval / indexing ──────────────────────────────────────────────────
    default_top_k: int = 5
    index_batch_size: int = 500
    embed_batch_size: int = 64

    # ── Derived fields (computed from repo root — not env vars) ───────────────
    # These use @computed_field so Pydantic includes them in .model_dump() etc.
    # but they are never read from the environment.

    @computed_field
    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @computed_field
    @property
    def content_list_path(self) -> Path:
        return (
            _REPO_ROOT
            / "data/outputs/annual_reports/adbe-2023-annual-report/auto"
            / "adbe-2023-annual-report_content_list.json"
        )

    @computed_field
    @property
    def images_base_dir(self) -> Path:
        """Directory where MinerU stores extracted images (sibling of the JSON)."""
        return self.content_list_path.parent / "images"

    @computed_field
    @property
    def chroma_db_path(self) -> str:
        return str(_REPO_ROOT / "data" / "chroma_db")

    @computed_field
    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # ── Startup validation ────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _warn_if_tracing_incomplete(self) -> "Settings":
        """Warn if only one of the two Langfuse keys is set (likely a misconfiguration)."""
        keys = (self.langfuse_public_key, self.langfuse_secret_key)
        if any(keys) and not all(keys):
            import warnings

            warnings.warn(
                "Only one Langfuse key is set — tracing will be disabled. "
                "Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable it.",
                stacklevel=2,
            )
        return self


# Module-level singleton — import this everywhere
# Pydantic raises ValidationError here if OPENAI_API_KEY is missing,
# which stops the process immediately with a clear message.
settings = Settings()
