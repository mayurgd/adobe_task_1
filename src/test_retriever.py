"""
test_retriever.py  —  interactively ask questions and retrieve chunks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retriever import query_documents

while True:
    query = input("\nEnter query (or 'quit'): ").strip()
    if query.lower() in ("quit", "q", "exit"):
        break
    if not query:
        continue

    results = query_documents(query, top_k=7)
    print()
    for r in results:
        print(f"[{r['rank']}] score={r['score']}  type={r['type']}  pages={r['pages']}")
        print(f"  heading : {r['heading']}")
        print(f"  text    : {r['text']}")
        print()
