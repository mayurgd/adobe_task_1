"""
streaming.py

Streaming helper for one agent turn.

Consumes ``agent.astream_events()`` (LangGraph v2 protocol) and handles:
  - Todo list display (``write_todos`` tool events)
  - Document retrieval progress (``answer_from_documents`` tool events)
  - Final reply extraction from the terminal ``LangGraph`` chain-end event

The caller (``agent.py``) is responsible for appending the returned reply to the
conversation history and for any Langfuse span wrapping around the full turn.

Public API:
    stream_agent_turn(agent, conversation) -> str
"""

from __future__ import annotations


async def stream_agent_turn(agent, conversation: list[dict]) -> str:
    """Stream one agent turn via ``astream_events`` (LangGraph v2).

    Prints live progress to stdout:
      - ``╔═ Todos ═╗ …`` whenever ``write_todos`` is called
      - ``[Searching documents] "…"`` at the start of each retrieval
      - ``[Retrieval complete]`` when retrieval finishes
      - ``Agent: …`` for the final synthesised answer

    Args:
        agent:        A LangGraph agent created by ``create_deep_agent()``.
        conversation: Full message history as a list of
                      ``{"role": "user"|"assistant", "content": "…"}`` dicts.

    Returns:
        The agent's complete reply string (empty string if no reply was found
        in the event stream).
    """
    active_retrieval_runs: set[str] = set()
    final_reply: str = ""

    async for event in agent.astream_events({"messages": conversation}, version="v2"):
        kind: str = event["event"]
        name: str = event.get("name", "")
        run_id: str = event.get("run_id", "")
        data: dict = event.get("data", {})

        # ── Todo list update ──────────────────────────────────────────────────
        if kind == "on_tool_start" and name == "write_todos":
            _print_todos(data)

        # ── Document retrieval start ──────────────────────────────────────────
        elif kind == "on_tool_start" and name == "answer_from_documents":
            active_retrieval_runs.add(run_id)
            query_str = _extract_query(data)
            print(f'\n[Searching documents] "{query_str}"', flush=True)

        # ── Document retrieval end ────────────────────────────────────────────
        elif kind == "on_tool_end" and name == "answer_from_documents":
            active_retrieval_runs.discard(run_id)
            print("[Retrieval complete]", flush=True)

        # ── Final graph state ─────────────────────────────────────────────────
        elif kind == "on_chain_end" and name == "LangGraph":
            final_reply = _extract_final_reply(data)

    if final_reply:
        print(f"\nAgent: {final_reply}")

    return final_reply


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_todos(data: dict) -> None:
    """Render the todo list from a ``write_todos`` tool-start event."""
    inp = data.get("input", {})
    todos_input = inp.get("todos", inp) if isinstance(inp, dict) else inp
    if not (isinstance(todos_input, list) and todos_input):
        return

    print("\n╔═ Todos ══════════════════════════════════", flush=True)
    for i, todo in enumerate(todos_input, 1):
        if isinstance(todo, dict):
            status = todo.get("status", "pending")
            title = todo.get("title") or todo.get("content") or str(todo)
            marker = "✓" if status in ("completed", "done") else "○"
            print(f"  {i}. {marker} {title}  [{status}]", flush=True)
        else:
            print(f"  {i}. • {todo}", flush=True)
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
