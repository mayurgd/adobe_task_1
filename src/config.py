"""
config.py

Single source of truth for environment variables and project constants.
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

    # ── Embedding / reranker ──────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── MinerU ────────────────────────────────────────────────────────────────
    mineru_output_dir: str = str(_REPO_ROOT / "data" / "docs" / "outputs")

    # ── Chunking ──────────────────────────────────────────────────────────────
    min_text_length: int = 30

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = 5
    retrieval_candidate_k: int = 15
    index_batch_size: int = 500
    embed_batch_size: int = 64

    # ── Derived paths ─────────────────────────────────────────────────────────
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
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @model_validator(mode="after")
    def _warn_if_tracing_incomplete(self) -> "Settings":
        keys = (self.langfuse_public_key, self.langfuse_secret_key)
        if any(keys) and not all(keys):
            import warnings

            warnings.warn(
                "Only one Langfuse key is set — tracing disabled. "
                "Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
                stacklevel=2,
            )
        return self


settings = Settings()
