from __future__ import annotations

import asyncio
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
from index_documents import format_chunks_as_context, query_documents  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

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

## Planning — MANDATORY for every request
Before calling any tools or composing your reply you MUST call `write_todos`
to lay out a numbered step-by-step plan for the full request.

Example plan for a multi-question input:
  1. Retrieve revenue trend data
  2. Retrieve underperforming-department data
  3. Retrieve key risks from last quarter
  4. Synthesise all findings into a structured reply

Update `write_todos` again (marking tasks done) as you complete each step.
This makes your reasoning transparent and traceable.

## Tool use — MANDATORY
You have access to the `answer_from_documents` tool which searches the indexed
company documents and returns relevant excerpts.

ALWAYS call `answer_from_documents` before composing your reply whenever the
user asks about:
  - financial figures, revenue, profit, costs, or KPIs
  - strategic priorities, goals, or initiatives
  - risks, challenges, or opportunities
  - operational performance or departmental results
  - any specific fact, date, name, or number from a company document

Do NOT answer from memory or general knowledge when document-specific
information is being requested. Plan first, retrieve second, then answer.

## Reflection — MANDATORY after every retrieval
After each `answer_from_documents` call you MUST critically evaluate the result
before moving on:
  1. Is the answer complete and specific enough for this sub-question?
  2. If there are gaps — refine the query and call `answer_from_documents`
     again with a more targeted question.
  3. If the answer is satisfactory — mark that `write_todos` step as
     "completed" and proceed to the next sub-question.

Keep iterating (retrieve → reflect → re-retrieve if needed) until every
part of the user's request is backed by solid evidence from the documents.
Only then compose the final response.

## Response guidelines
  - Be concise, factual, and data-driven.
  - Cite specific figures, percentages, page numbers, and named entities from
    the retrieved excerpts whenever available.
  - If the retrieved excerpts do not contain enough information to answer
    confidently, say so clearly — do not hallucinate.
  - Structure longer responses with bullet points or short sections.
  - Maintain context across the full conversation history.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────────────────────────────────────


@tool
def answer_from_documents(query: str) -> str:
    """Retrieve relevant passages from the indexed company documents and generate
    a factual, source-grounded answer.

    Use this tool whenever the user asks about specific figures, financial results,
    strategies, risks, or any content that may be found in the uploaded company
    documents (annual reports, quarterly reports, etc.).

    Args:
        query: The user's question expressed in natural language.

    Returns:
        A concise answer grounded in the retrieved document excerpts, with
        references to headings and page numbers where available.
    """
    # Obtain a Langfuse client for nested tracing when credentials are present.
    # get_client() inherits the active trace context (set by the outer agent-turn
    # span in run()), so observations created here nest automatically.
    _lf = None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client as _lf_get_client  # type: ignore

            _lf = _lf_get_client()
        except (ImportError, Exception):
            pass

    # ── Retrieval step ────────────────────────────────────────────────────────
    try:
        if _lf:
            with _lf.start_as_current_observation(
                as_type="span",
                name="document-retrieval",
                input={"query": query, "top_k": 5},
            ) as retrieval_obs:
                chunks = query_documents(query, top_k=5)
                retrieval_obs.update(
                    output={
                        "chunks_retrieved": len(chunks),
                        "headings": [c.get("heading", "") for c in chunks],
                    }
                )
        else:
            chunks = query_documents(query, top_k=5)
    except Exception as exc:
        return f"[retrieval error] Could not query the document index: {exc}"

    if not chunks:
        return "No relevant content found in the document index for this query."

    context = format_chunks_as_context(chunks)

    rag_prompt = (
        "You are a financial analyst assistant. Using ONLY the document excerpts "
        "provided below, answer the question. Be concise and cite specific figures "
        "and page numbers where available. If the answer cannot be determined from "
        "the excerpts, say so explicitly — do not hallucinate.\n\n"
        f"Question: {query}\n\n"
        f"Document excerpts:\n{context}"
    )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    # ── RAG generation step ───────────────────────────────────────────────────
    if _lf:
        with _lf.start_as_current_observation(
            as_type="generation",
            name="rag-generation",
            model=model_name,
            input=rag_prompt,
        ) as rag_gen:
            response = llm.invoke(rag_prompt)
            rag_gen.update(output=response.content)
    else:
        response = llm.invoke(rag_prompt)

    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list = [answer_from_documents]


# ─────────────────────────────────────────────────────────────────────────────
# Langfuse helper
# ─────────────────────────────────────────────────────────────────────────────


def _get_langfuse():
    """
    Return a Langfuse client when credentials exist, otherwise None.
    Uses the new get_client() API (langfuse >= 2.x).
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not (public_key and secret_key):
        return None  # credentials not configured — tracing disabled

    try:
        from langfuse import get_client  # type: ignore

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        # get_client() reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
        # LANGFUSE_HOST from the environment automatically.
        client = get_client()
        print(f"[info] Langfuse tracing enabled → {host}")
        return client

    except ImportError:
        print(
            "[warn] langfuse package not found — tracing disabled.\n"
            "       Install it with:  pip install langfuse"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Streaming helper
# ─────────────────────────────────────────────────────────────────────────────


async def _stream_agent_turn(agent, conversation: list[dict]) -> str:
    """Stream one agent turn via astream_events.

    Displays the full ReAct loop in the terminal:
      [Agent's plan]      — free-text reasoning before the first tool call
      ╔═ Agent plan ═╗    — write_todos planning step (numbered + status)
      [Searching …]       — each answer_from_documents invocation
      [Retrieval complete]
      ╔═ Reflection N ╗   — LLM reasoning between tool calls (feedback loop)
      Agent: …            — final synthesised response

    Returns the complete reply string for appending to conversation history.
    """
    active_retrieval_runs: set[str] = set()
    final_reply: str = ""

    # Tokens emitted before the first tool call (free-text plan / reasoning).
    pre_tool_tokens: list[str] = []
    first_tool_seen: bool = False

    # Tokens emitted between tool calls — buffered so we can label them as
    # ╔═ Reflection N ╗ when a subsequent tool fires, or as the final
    # answer when the graph ends.  This implements the observe → reflect
    # → act feedback loop described in the ReAct architecture.
    current_llm_section: list[str] = []
    any_tool_completed: bool = False
    reflection_count: int = 0

    def _flush_section_as_reflection() -> None:
        nonlocal reflection_count
        text = "".join(current_llm_section).strip()
        current_llm_section.clear()
        if not text:
            return
        reflection_count += 1
        print(
            f"\n╔═ Reflection {reflection_count} ══════════════════════════════",
            flush=True,
        )
        for line in text.splitlines():
            print(f"  {line}", flush=True)
        print("╚══════════════════════════════════════════", flush=True)

    async for event in agent.astream_events({"messages": conversation}, version="v2"):
        kind: str = event["event"]
        name: str = event.get("name", "")
        run_id: str = event.get("run_id", "")
        data: dict = event.get("data", {})

        # ── All on_tool_start events ─────────────────────────────────────────
        if kind == "on_tool_start":
            if not first_tool_seen:
                # Very first tool: flush any free-text pre-tool reasoning.
                first_tool_seen = True
                if pre_tool_tokens:
                    reasoning = "".join(pre_tool_tokens).strip()
                    pre_tool_tokens.clear()
                    if reasoning:
                        print("\n[Agent's plan]", flush=True)
                        for line in reasoning.splitlines():
                            print(f"  {line}", flush=True)
            elif any_tool_completed and current_llm_section:
                # A new tool starts after a previous retrieval: the buffered
                # LLM tokens are the agent's reflection on what it observed.
                _flush_section_as_reflection()

            # ── Planning / todo display ───────────────────────────────────────
            if name == "write_todos":
                inp = data.get("input", {})
                todos = inp.get("todos", inp) if isinstance(inp, dict) else inp
                print("\n╔═ Agent plan ══════════════════════════════", flush=True)
                if isinstance(todos, list):
                    for i, todo in enumerate(todos, 1):
                        if isinstance(todo, dict):
                            status = todo.get("status", "pending")
                            title = todo.get("title", str(todo))
                            marker = (
                                "\u2713"
                                if status in ("completed", "done")
                                else "\u25cb"
                            )
                            print(f"  {i}. {marker} {title}  [{status}]", flush=True)
                        else:
                            print(f"  {i}. \u2022 {todo}", flush=True)
                else:
                    print(f"  {todos}", flush=True)
                print("╚══════════════════════════════════════════", flush=True)

            # ── Document retrieval display ────────────────────────────────────
            elif name == "answer_from_documents":
                active_retrieval_runs.add(run_id)
                inp = data.get("input", {})
                query_str = (
                    inp.get("query", str(inp)) if isinstance(inp, dict) else str(inp)
                )
                print(f'\n[Searching documents] "{query_str}"', flush=True)

        # ── Tool end ──────────────────────────────────────────────────────────
        elif kind == "on_tool_end" and name == "answer_from_documents":
            active_retrieval_runs.discard(run_id)
            any_tool_completed = True
            print("[Retrieval complete]", flush=True)

        # ── LLM token stream ──────────────────────────────────────────────────
        # Inner RAG LLM tokens (inside answer_from_documents) are suppressed;
        # only outer agent LLM tokens are captured.
        elif kind == "on_chat_model_stream" and not active_retrieval_runs:
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                if not first_tool_seen:
                    # Pre-tool free-text reasoning.
                    pre_tool_tokens.append(chunk.content)
                else:
                    # Post-tool tokens: buffered — we don't yet know if this
                    # is reflection (more tools follow) or the final answer.
                    current_llm_section.append(chunk.content)

        # ── Final graph state ─────────────────────────────────────────────────
        elif kind == "on_chain_end" and name == "LangGraph":
            output = data.get("output", {})
            messages = output.get("messages", [])
            if messages:
                last = messages[-1]
                final_reply = last.content if hasattr(last, "content") else str(last)

    # The last buffered LLM section is the agent's final answer (no subsequent
    # tool call fired to flush it as reflection).  Print it cleanly.
    answer = final_reply or "".join(current_llm_section)
    if answer:
        print(f"\nAgent: {answer}")

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Interactive conversation loop
# ─────────────────────────────────────────────────────────────────────────────


async def run() -> None:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(model=model_name, temperature=0)

    agent = create_deep_agent(
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        model=llm,
    )

    langfuse = _get_langfuse()

    # Full message history — passed on every turn so the agent has context
    conversation: list[dict[str, str]] = []

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║      AI Leadership Insight Agent             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Model  : {model_name}")
    print(
        f"  Tracing: {'Langfuse enabled' if langfuse else 'disabled (set LANGFUSE_* vars)'}"
    )
    print()
    print("  Ask any question about company performance, strategy, or risk.")
    print("  Type 'exit' or 'quit' (or Ctrl-C) to stop.\n")

    try:
        while True:
            # ── Read input ───────────────────────────────────────────────────
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

            # ── Append user turn ─────────────────────────────────────────────
            conversation.append({"role": "user", "content": raw})

            # ── Stream agent turn under a Langfuse trace span ─────────────────
            try:
                if langfuse:
                    with langfuse.start_as_current_observation(
                        as_type="span",
                        name="agent-turn",
                        input=raw,
                    ) as span:
                        with langfuse.start_as_current_observation(
                            as_type="span",
                            name="agent-invocation",
                            input=raw,
                        ) as invocation:
                            reply = await _stream_agent_turn(agent, conversation)
                            invocation.update(output=reply)
                        span.update(output=reply)
                else:
                    reply = await _stream_agent_turn(agent, conversation)
            except Exception as exc:
                print(f"\n[error] Agent call failed: {exc}\n")
                conversation.pop()  # roll back the failed user message
                continue

            # ── Append assistant turn ─────────────────────────────────────────
            conversation.append({"role": "assistant", "content": reply})

    finally:
        # Flush any buffered Langfuse events before the process exits
        if langfuse:
            langfuse.flush()


if __name__ == "__main__":
    asyncio.run(run())
