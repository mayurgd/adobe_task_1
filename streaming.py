"""
streaming.py

Streaming helper for one agent turn.

Consumes ``agent.astream_events()`` (LangGraph v2 protocol) and handles:
  - Todo list display / SSE emission  (``write_todos`` tool events)
  - Document retrieval progress       (``answer_from_documents`` tool events)
  - Final reply extraction from the terminal ``LangGraph`` chain-end event

Two modes:
  1. CLI mode  — ``event_queue=None``  → prints to stdout (original behaviour)
  2. SSE mode  — ``event_queue``       → puts structured dicts into the queue
                                         for app.py to forward to the browser

Public API:
    stream_agent_turn(agent, conversation, event_queue=None) -> str
"""

from __future__ import annotations

import asyncio
import json


async def stream_agent_turn(
    agent,
    conversation: list[dict],
    event_queue: asyncio.Queue | None = None,
) -> str:
    """Stream one agent turn via ``astream_events`` (LangGraph v2).

    When *event_queue* is ``None`` (CLI mode) progress is printed to stdout.
    When *event_queue* is provided (SSE / HTTP mode) every status update is
    put onto the queue as a dict:

        {"type": "todos",      "todos": [...]}
        {"type": "searching",  "query": "..."}
        {"type": "retrieved",  "query": "..."}   ← query echoed back for UI matching
        {"type": "answer",     "answer": "...", "sources": [...]}
        {"type": "error",      "message": "..."}

    The caller in app.py drains the queue and serialises each dict as an
    SSE ``data:`` line.

    Args:
        agent:        A LangGraph agent created by ``create_deep_agent()``.
        conversation: Full message history as a list of
                      ``{"role": "user"|"assistant", "content": "…"}`` dicts.
        event_queue:  Optional asyncio.Queue for SSE mode.

    Returns:
        The agent's complete reply string (empty string if no reply was found
        in the event stream).
    """
    # Maps run_id → query string so on_tool_end can echo the right query back
    _active_queries: dict[str, str] = {}

    final_reply: str = ""

    async for event in agent.astream_events({"messages": conversation}, version="v2"):
        kind: str = event["event"]
        name: str = event.get("name", "")
        run_id: str = event.get("run_id", "")
        data: dict = event.get("data", {})

        # ── Todo list update ──────────────────────────────────────────────────
        if kind == "on_tool_start" and name == "write_todos":
            todos = _parse_todos(data)
            if event_queue:
                await event_queue.put({"type": "todos", "todos": todos})
            else:
                _print_todos_cli(todos)

        # ── Document retrieval start ──────────────────────────────────────────
        elif kind == "on_tool_start" and name == "answer_from_documents":
            query_str = _extract_query(data)
            # Store so the paired on_tool_end can echo it back
            _active_queries[run_id] = query_str
            if event_queue:
                await event_queue.put({"type": "searching", "query": query_str})
            else:
                print(f'\n[Searching documents] "{query_str}"', flush=True)

        # ── Document retrieval end ────────────────────────────────────────────
        elif kind == "on_tool_end" and name == "answer_from_documents":
            # Echo the same query string so the frontend can match by query,
            # not by insertion order — fixes the "first item stays spinning"
            # bug that occurs when the second search starts before the first
            # retrieved event arrives.
            query_str = _active_queries.pop(run_id, "")
            if event_queue:
                await event_queue.put({"type": "retrieved", "query": query_str})
            else:
                print("[Retrieval complete]", flush=True)

        # ── Final graph state ─────────────────────────────────────────────────
        elif kind == "on_chain_end" and name == "LangGraph":
            final_reply = _extract_final_reply(data)

    if event_queue is None and final_reply:
        print(f"\nAgent: {final_reply}")

    return final_reply


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_todos(data: dict) -> list[dict]:
    """Extract a normalised list of todo dicts from a ``write_todos`` event."""
    inp = data.get("input", {})
    todos_raw = inp.get("todos", inp) if isinstance(inp, dict) else inp
    if not isinstance(todos_raw, list):
        return []

    result = []
    for todo in todos_raw:
        if isinstance(todo, dict):
            result.append(
                {
                    "title": todo.get("title") or todo.get("content") or str(todo),
                    "status": todo.get("status", "pending"),
                }
            )
        else:
            result.append({"title": str(todo), "status": "pending"})
    return result


def _print_todos_cli(todos: list[dict]) -> None:
    """Render a todo list to stdout (CLI mode)."""
    if not todos:
        return
    print("\n╔═ Todos ══════════════════════════════════", flush=True)
    for i, todo in enumerate(todos, 1):
        status = todo.get("status", "pending")
        marker = "✓" if status in ("completed", "done") else "○"
        print(f"  {i}. {marker} {todo['title']}  [{status}]", flush=True)
    print("╚══════════════════════════════════════════", flush=True)


def _extract_query(data: dict) -> str:
    """Pull the query string from an ``answer_from_documents`` tool-start event."""
    inp = data.get("input", {})
    if isinstance(inp, dict):
        return inp.get("query", str(inp))
    return str(inp)


def _extract_final_reply(data: dict) -> str:
    """Extract the last assistant message content from a LangGraph chain-end event."""
    output = data.get("output", {})
    messages = output.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    return last.content if hasattr(last, "content") else str(last)
