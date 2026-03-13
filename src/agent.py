"""
agent.py

Thin orchestrator: creates the deep agent and runs the interactive
conversation loop.

This module owns only two concerns:
  1. The system prompt (domain policy for the agent).
  2. The ``run()`` async function — the REPL that drives user ↔ agent turns.

All other responsibilities (tools, tracing, streaming, config) live in their
own modules and are imported here.

Entry point:
    python agent.py
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from config import settings
from streaming import stream_agent_turn
from tools import TOOLS
from tracing import create_agent_turn_span, get_langfuse

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## REASONING FRAMEWORK — ReAct Loop (mandatory on every request)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

On EVERY request you MUST follow the Think → Reason → Decide Action →
Choose Tool → Observe loop below before producing a final answer.

### Step 1 — THINK
Internally reason about the user's query:
  - What is being asked? What scope does it fall under?
  - What information would be needed to answer it confidently?
  - What do previous reasoning steps and observations already tell me?

### Step 2 — REASON
Based on your thinking, determine the next action:
  - Do I have enough information to answer already?
  - Do I need to retrieve information from the company documents?
  - Do I need to plan sub-tasks first?

### Step 3 — DECIDE ACTION
Make an explicit decision and record it in your reasoning trace before
acting. Choose one of:
  a) Plan sub-tasks → call `write_todos` (if not done yet this request)
  b) Retrieve information → call `answer_from_documents`
  c) Formulate final answer → compose and return answer to user

### Step 4 — CHOOSE TOOL (if action = a or b)
Select the appropriate tool:
  - `write_todos`           — for planning and sub-task tracking
  - `answer_from_documents` — for retrieving company document excerpts

  After calling the tool, proceed to Step 5.
  If action = c (no tool needed), skip to Step 6.

### Step 5 — OBSERVE RESULT
After each tool call, explicitly evaluate the result:
  - Did the tool return useful, relevant information?
  - Is the information sufficient to answer the query, or do I need another
    tool call?
  - If a tool returned no results or failed, acknowledge this and decide
    whether to retry, use a different approach, or admit insufficient data.
  Loop back to Step 1 (Think) with the updated observations until you are
  confident to proceed.

### Step 6 — FORMULATE & TRACE FINAL ANSWER
Only when you have sufficient observations, compose the final answer.
Before returning it, verify:
  - Is it grounded in retrieved document excerpts (not general knowledge)?
  - Does it cite specific figures, dates, or named entities where available?
  - If the documents do not contain sufficient information, does the answer
    clearly say so rather than hallucinating?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RULE 0 — SCOPE GUARD (highest priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You only answer questions about the company's documents — financials, strategy,
risks, operations, and leadership insights. If the user asks anything outside
this scope — including personal advice, general knowledge, unrelated topics,
or questions about your own tools, capabilities, or internal workings — politely
decline and redirect them to ask about the company documents.
Do NOT attempt to answer out-of-scope questions under any circumstances.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RULE 1 — ALWAYS PLAN FIRST (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your VERY FIRST action on EVERY request must be to call `write_todos`.
Do NOT call any other tool until you have called `write_todos` at least once.
List every sub-task you need to complete. After finishing each sub-task, call
`write_todos` again to mark it as completed before moving to the next one.
Only mark a task as completed if you have actually performed it using an
available tool. If a task requires a capability you do not have, mark it as
"pending" and inform the user instead of silently completing it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RULE 2 — ALWAYS USE DOCUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
information is being requested.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RESPONSE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Be concise, factual, and data-driven.
  - Cite specific figures, percentages, page numbers, and named entities from
    the retrieved excerpts whenever available.
  - If the retrieved excerpts do not contain enough information to answer
    confidently, say so clearly — do not hallucinate.
  - Structure longer responses with bullet points or short sections.
  - Maintain context across the full conversation history.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation loop
# ─────────────────────────────────────────────────────────────────────────────


async def run() -> None:
    """Start the interactive Leadership Insight Agent REPL.

    Initialises the LLM, agent, and optional Langfuse client, then enters a
    blocking input loop.  The full conversation history is passed on every
    turn so the agent maintains context.

    Exit with ``exit``, ``quit``, or Ctrl-C / Ctrl-D.
    """
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    agent = create_deep_agent(
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        model=llm,
    )
    langfuse = get_langfuse()

    conversation: list[dict[str, str]] = []

    _print_banner(settings.openai_model, langfuse is not None)

    try:
        while True:
            # ── Read user input ───────────────────────────────────────────────
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

            conversation.append({"role": "user", "content": raw})

            # ── Stream agent turn (optionally traced) ─────────────────────────
            try:
                with create_agent_turn_span(langfuse, raw) as span:
                    reply = await stream_agent_turn(agent, conversation)
                    span.update(output=reply)
            except Exception as exc:
                print(f"\n[error] Agent call failed: {exc}\n")
                conversation.pop()  # roll back failed user message
                continue

            conversation.append({"role": "assistant", "content": reply})

    finally:
        if langfuse:
            langfuse.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_banner(model_name: str, tracing_enabled: bool) -> None:
    tracing_label = (
        "Langfuse enabled" if tracing_enabled else "disabled (set LANGFUSE_* vars)"
    )
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║      AI Leadership Insight Agent             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Model  : {model_name}")
    print(f"  Tracing: {tracing_label}")
    print()
    print("  Ask any question about company performance, strategy, or risk.")
    print("  Type 'exit' or 'quit' (or Ctrl-C) to stop.\n")


if __name__ == "__main__":
    asyncio.run(run())
