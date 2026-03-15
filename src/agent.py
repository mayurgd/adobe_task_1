"""
agent.py — AI Leadership Insight Agent
Run: python agent.py
Deps: pip install rich langchain langchain-openai
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import settings
from kb_setup.answer_query import answer_query
from kb_setup.doc_registry import collection_name_for, list_all

console = Console()

COLLECTION_NAME = collection_name_for("d164e2f9-27b8-4963-a1ae-85e506a4c762_q12025.pdf")


# ── Tool ──────────────────────────────────────────────────────────────────────


@tool
def answer_from_documents(query: str) -> str:
    """
    Retrieve a CFO-grade answer from indexed company documents.
    Use for any question about financial performance, strategy, targets, or risk.
    """
    result = answer_query(query, collection_name=COLLECTION_NAME)
    lines = [result["answer"], "", "Cited sources:"]
    for s in result["cited_sources"]:
        lines.append(
            f"  [Source {s['number']}] {s['heading']} | pages {s['pages']} | score {s['score']:.3f}"
        )
    return "\n".join(lines)


# ── Observability helpers ─────────────────────────────────────────────────────


def _render_todos(todos: list[dict]) -> None:
    if not todos:
        return
    completed = sum(1 for t in todos if t.get("status") == "completed")
    total = len(todos)
    pct = int((completed / total) * 100) if total else 0
    bar = "[green]" + "█" * int(pct / 5) + "[/green]" + "░" * (20 - int(pct / 5))

    table = Table.grid(padding=(0, 1))
    for t in todos:
        s = t.get("status", "pending")
        icon = (
            "[green]✓[/green]"
            if s == "completed"
            else ("[yellow]◉[/yellow]" if s == "in_progress" else "[dim]○[/dim]")
        )
        style = (
            "dim strike"
            if s == "completed"
            else ("yellow" if s == "in_progress" else "")
        )
        label = t.get("title") or t.get("content") or str(t)
        table.add_row(icon, Text(label, style=style))

    console.print(
        Panel(
            table,
            title=f"[bold]todos[/bold]  {bar}  {completed}/{total}",
            border_style="bright_black",
            padding=(0, 1),
        )
    )


def _parse_todos(data: dict) -> list[dict]:
    inp = data.get("input", {})
    raw = inp.get("todos", inp) if isinstance(inp, dict) else inp
    if not isinstance(raw, list):
        return []
    result = []
    for t in raw:
        if isinstance(t, dict):
            result.append(
                {
                    "title": t.get("title") or t.get("content") or str(t),
                    "status": t.get("status", "pending"),
                }
            )
        else:
            result.append({"title": str(t), "status": "pending"})
    return result


def _extract_query(data: dict) -> str:
    inp = data.get("input", {})
    return inp.get("query", str(inp)) if isinstance(inp, dict) else str(inp)


def _extract_final_reply(data: dict) -> str:
    messages = data.get("output", {}).get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    return last.content if hasattr(last, "content") else str(last)


# ── Streaming ─────────────────────────────────────────────────────────────────


async def stream_agent_turn(agent, conversation: list[dict]) -> str:
    final_reply = ""

    async for event in agent.astream_events({"messages": conversation}, version="v2"):
        kind = event["event"]
        name = event.get("name", "")
        data = event.get("data", {})

        if kind == "on_tool_start" and name == "write_todos":
            _render_todos(_parse_todos(data))

        elif kind == "on_tool_start" and name == "answer_from_documents":
            query = _extract_query(data)
            console.print(f'\n[cyan]→ searching:[/cyan] [dim]"{query}"[/dim]')

        elif kind == "on_tool_end" and name == "answer_from_documents":
            console.print("[green]← retrieval complete[/green]")

        elif kind == "on_chain_end" and name == "LangGraph":
            final_reply = _extract_final_reply(data)

    return final_reply


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an Insight Agent that analyses documents

On every request:
1. Use write_todos to plan your steps before starting.
2. Call answer_from_documents to retrieve relevant excerpts.
3. Synthesise a grounded, cited final answer. Never hallucinate.
"""


# ── REPL ──────────────────────────────────────────────────────────────────────


async def run() -> None:
    agent = create_agent(
        model=settings.openai_model,
        tools=[answer_from_documents],
        middleware=[TodoListMiddleware(system_prompt=SYSTEM_PROMPT)],
    )
    conversation: list[dict] = []

    docs = list_all()
    doc_lines = (
        "\n".join(f"  • {d['filename']} ({d['chunk_count']} chunks)" for d in docs)
        or "  (none — run index_documents.py first)"
    )
    console.print(
        Panel(
            f"[bold]Insight Agent[/bold]  [dim]{settings.openai_model}[/dim]\n\n"
            f"{doc_lines}\n\n[dim]exit / quit / Ctrl-C to stop[/dim]",
            border_style="bright_black",
        )
    )

    while True:
        try:
            raw = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not raw or raw.lower() in {"exit", "quit"}:
            break

        conversation.append({"role": "user", "content": raw})

        try:
            reply = await stream_agent_turn(agent, conversation)
        except Exception as exc:
            console.print(f"[red]error: {exc}[/red]")
            conversation.pop()
            continue

        if reply:
            console.print(f"\n[bold]Agent:[/bold] {reply}")
        conversation.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    asyncio.run(run())
