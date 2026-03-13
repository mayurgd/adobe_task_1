from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from the repo root (one level up from src/)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Guard: OpenAI key must be present before any heavy imports ────────────────
if not os.getenv("OPENAI_API_KEY"):
    sys.exit(
        "[error] OPENAI_API_KEY is not set.\n"
        "Copy .env.example to .env and add your key, or export it in your shell."
    )

from deepagents import create_deep_agent  # noqa: E402  (after env check)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an AI Leadership Insight Agent specialised in analysing company
documents — annual reports, quarterly reports, strategy notes, and operational
updates.

Your job is to help executives and leadership teams answer questions such as:
  • "What is our current revenue trend?"
  • "Which departments are underperforming?"
  • "What were the key risks highlighted in the last quarter?"
  • "Summarise the strategic priorities for the next fiscal year."

Guidelines:
  - Be concise, factual, and data-driven.
  - Cite specific figures, percentages, and named entities when available.
  - If information is not available or uncertain, say so clearly — do not hallucinate.
  - Structure longer responses with bullet points or short sections.
  - Maintain context across the full conversation history.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry  (add document-retrieval tools here when ready)
# ─────────────────────────────────────────────────────────────────────────────
# Example — uncomment and implement once ChromaDB index is ready:
#
# from tools.retrieval import search_documents, fetch_table, fetch_image
# TOOLS = [search_documents, fetch_table, fetch_image]

TOOLS: list = []  # no tools wired in yet


# ─────────────────────────────────────────────────────────────────────────────
# Langfuse helper
# ─────────────────────────────────────────────────────────────────────────────


def _build_langfuse_handler():
    """
    Return a Langfuse CallbackHandler when credentials exist, otherwise None.
    Deferred import so a missing langfuse install only raises a soft warning.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not (public_key and secret_key):
        return None  # credentials not configured — tracing disabled

    try:
        from langfuse.callback import CallbackHandler  # type: ignore

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        print(f"[info] Langfuse tracing enabled → {host}")
        return handler

    except ImportError:
        print(
            "[warn] langfuse package not found — tracing disabled.\n"
            "       Install it with:  pip install langfuse"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Interactive conversation loop
# ─────────────────────────────────────────────────────────────────────────────


def run() -> None:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    agent = create_deep_agent(
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    langfuse_handler = _build_langfuse_handler()

    # Full message history — passed on every turn so the agent has context
    conversation: list[dict[str, str]] = []

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║      AI Leadership Insight Agent             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Model  : {model_name}")
    print(
        f"  Tracing: {'Langfuse enabled' if langfuse_handler else 'disabled (set LANGFUSE_* vars)'}"
    )
    print()
    print("  Ask any question about company performance, strategy, or risk.")
    print("  Type 'exit' or 'quit' (or Ctrl-C) to stop.\n")

    while True:
        # ── Read input ───────────────────────────────────────────────────────
        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # ── Append user turn ─────────────────────────────────────────────────
        conversation.append({"role": "user", "content": raw})

        # ── Invoke agent ─────────────────────────────────────────────────────
        invoke_config: dict = {}
        if langfuse_handler:
            invoke_config["callbacks"] = [langfuse_handler]

        try:
            state = agent.invoke(
                {"messages": conversation},
                config=invoke_config,
            )
        except Exception as exc:
            print(f"\n[error] Agent call failed: {exc}\n")
            conversation.pop()  # roll back the failed user message
            continue

        # ── Extract AI reply ─────────────────────────────────────────────────
        last_message = state["messages"][-1]
        reply: str = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # ── Append assistant turn and print ──────────────────────────────────
        conversation.append({"role": "assistant", "content": reply})
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    run()
