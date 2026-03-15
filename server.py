"""
server.py — Insight Agent FastAPI backend

Mirrors the pipeline from app.py exactly:
  - MinerU via subprocess (same cmd as app.py)
  - Same output directory structure: DOCS_OUTPUT_DIR/<stem>/auto/<stem>_content_list.json
  - Registry persistence to data/registry.json (survives server restarts)
  - Per-document ChromaDB collections via kb_setup.doc_registry
  - SSE /ask with live todo + searching + retrieved + answer events

Endpoints:
  POST /ingest            Upload file → background MinerU pipeline
  GET  /status/{doc_id}  Poll indexing status
  GET  /docs-list         Session rehydration (merges registry + in-memory jobs)
  GET  /file/{doc_id}     Serve raw PDF to the viewer
  POST /ask               SSE stream of agent progress events
  GET  /health            Quick health check
  GET  /                  Serves index.html

Run:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Project path setup ────────────────────────────────────────────────────────
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from config import settings
from kb_setup.answer_query import answer_query
from kb_setup.chunker import build_chunks
from kb_setup.doc_registry import collection_name_for, list_all, register
from kb_setup.indexer import load_embedding_model

# ── Paths (mirrors app.py exactly) ───────────────────────────────────────────
DOCS_INPUT_DIR = ROOT / "data" / "docs" / "inputs"
DOCS_OUTPUT_DIR = Path(settings.mineru_output_dir)  # data/docs/outputs
REGISTRY_PATH = ROOT / "data" / "registry.json"

DOCS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Insight Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ───────────────────────────────────────────────────────────
# doc_registry: { doc_id: { name, path, pages, collection_name } }
# job_status:   { doc_id: { status, message } }
doc_registry: dict[str, dict] = {}
job_status: dict[str, dict] = {}

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = load_embedding_model()
    return _embed_model


# ── Registry persistence (same as app.py) ────────────────────────────────────


def _load_registry() -> None:
    if not REGISTRY_PATH.exists():
        return
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        raw_registry: dict = data.get("doc_registry", {})
        raw_status: dict = data.get("job_status", {})

        for doc_id, entry in raw_registry.items():
            path = Path(entry["path"])
            if not path.exists():
                print(f"[registry] Skipping {doc_id[:8]}: file missing at {path}")
                continue
            doc_registry[doc_id] = {**entry, "path": path}
            doc_registry[doc_id].setdefault(
                "collection_name", collection_name_for(entry["name"])
            )
            js = raw_status.get(doc_id, {"status": "indexed", "message": "Restored."})
            # Anything that was mid-flight when server died → error
            if js["status"] == "processing":
                js = {
                    "status": "error",
                    "message": "Server restarted during indexing — please re-upload.",
                }
            job_status[doc_id] = js

        print(f"[registry] Restored {len(doc_registry)} document(s).")
    except Exception as exc:
        print(f"[registry] Load failed: {exc}")


def _save_registry() -> None:
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ser_registry = {
            doc_id: {**entry, "path": str(entry["path"])}
            for doc_id, entry in doc_registry.items()
        }
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"doc_registry": ser_registry, "job_status": job_status},
                f,
                indent=2,
            )
    except Exception as exc:
        print(f"[registry] Save failed: {exc}")


# ── Status helper ─────────────────────────────────────────────────────────────


def _set_status(doc_id: str, status: str, message: str) -> None:
    job_status[doc_id] = {"status": status, "message": message}
    print(f"[pipeline] [{status.upper()}] {doc_id[:8]} — {message}")
    if status in ("indexed", "error"):
        _save_registry()


# ── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _startup() -> None:
    print("[startup] Loading registry …")
    _load_registry()
    print("[startup] Warming up embedding model …")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_embed_model)
    print("[startup] Ready.")


# ── Background pipeline (mirrors app.py _run_pipeline exactly) ───────────────


def _run_pipeline(doc_id: str, file_path: Path) -> None:
    """
    Runs in a thread-pool worker via BackgroundTasks.

    Steps:
      1. mineru subprocess  →  <stem>/auto/<stem>_content_list.json
      2. build_chunks
      3. embed + upsert into per-doc ChromaDB collection
      4. register in kb_setup doc_registry
    """
    original_name = doc_registry[doc_id]["name"]
    coll_name = collection_name_for(original_name)

    try:
        # ── Step 1: MinerU ────────────────────────────────────────────────────
        _set_status(doc_id, "processing", "Running MinerU parser …")

        cmd = [
            "mineru",
            "-p",
            str(file_path),
            "-o",
            str(DOCS_OUTPUT_DIR),
            "-b",
            "pipeline",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU exited {result.returncode}.\nstderr: {result.stderr[-800:]}"
            )

        # MinerU writes to: <output_dir>/<stem>/auto/<stem>_content_list.json
        stem = file_path.stem
        content_list_path = (
            DOCS_OUTPUT_DIR / stem / "auto" / f"{stem}_content_list.json"
        )
        if not content_list_path.exists():
            raise FileNotFoundError(
                f"MinerU output not found at: {content_list_path}\n"
                f"stdout: {result.stdout[-400:]}"
            )

        # ── Step 2: Chunk ─────────────────────────────────────────────────────
        _set_status(doc_id, "processing", "Chunking document …")
        with open(content_list_path, encoding="utf-8") as f:
            content = json.load(f)

        chunks = build_chunks(content)
        if not chunks:
            raise ValueError(
                "No chunks produced — document may be empty or unreadable."
            )

        # ── Step 3: Embed + index into per-doc ChromaDB collection ────────────
        _set_status(doc_id, "processing", f"Embedding {len(chunks)} chunks …")

        model = _get_embed_model()
        embed_texts = [c["embed_text"] for c in chunks]
        embeddings = model.encode(
            embed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=settings.embed_batch_size,
        ).tolist()

        import chromadb as _chromadb

        client = _chromadb.PersistentClient(path=settings.chroma_db_path)

        # Always recreate so re-uploads are clean
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [c["id"] for c in chunks]
        metadatas = [
            {
                "type": c["type"],
                "heading": c["heading"],
                "text": c["text"],
                "source_idx_start": c.get("source_idx_start", c.get("source_idx", -1)),
                "source_idx": c.get("source_idx", -1),
                "img_path": c.get("img_path", ""),
                "caption": c.get("caption", ""),
                "pages": json.dumps(c["pages"]),
                "tables": json.dumps(c.get("tables", [])),
                "images": json.dumps(c.get("images", [])),
            }
            for c in chunks
        ]

        batch_size = settings.index_batch_size
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=embed_texts[i:end],
            )
            _set_status(doc_id, "processing", f"Storing chunks … {end}/{len(chunks)}")

        # ── Step 4: Register in kb_setup doc_registry ─────────────────────────
        register(original_name, coll_name, len(chunks))

        # Update page count from chunks (0-indexed pages → +1)
        all_pages = {p for c in chunks for p in c.get("pages", [])}
        if all_pages:
            doc_registry[doc_id]["pages"] = max(all_pages) + 1
        doc_registry[doc_id]["collection_name"] = coll_name

        _set_status(doc_id, "indexed", f"Indexed {len(chunks)} chunks successfully.")

    except subprocess.TimeoutExpired:
        _set_status(doc_id, "error", "MinerU timed out after 10 minutes.")
    except Exception as exc:
        _set_status(doc_id, "error", str(exc)[:500])


# ── SSE helper ────────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── /ask — SSE stream with live todos ────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str]
    selected_doc_id: str | None = None  # whichever doc is open in the viewer


def _resolve_target(
    doc_ids: list[str], selected_doc_id: str | None
) -> tuple[str | None, str]:
    """Return (target_id, error_message). error_message is '' if ok."""
    target_id = selected_doc_id if selected_doc_id in doc_registry else None
    if target_id is None:
        target_id = next(
            (d for d in doc_ids if job_status.get(d, {}).get("status") == "indexed"),
            None,
        )
    if not target_id:
        return None, "No indexed document found."
    js = job_status.get(target_id, {})
    if js.get("status") != "indexed":
        name = doc_registry.get(target_id, {}).get("name", target_id)
        return None, f"'{name}' is still {js.get('status', 'unknown')}. Please wait."
    return target_id, ""


def _build_sources(
    cited_sources: list[dict], target_id: str, coll_name: str
) -> list[dict]:
    """
    Convert answer_query cited_sources into UI source refs.
    Enriches each with source_idx_start + source_idx from ChromaDB metadata
    so the frontend can highlight the full bounding box range of the chunk.
    """
    try:
        import chromadb as _chromadb

        _client = _chromadb.PersistentClient(path=settings.chroma_db_path)
        _coll = _client.get_collection(coll_name)
        for cs in cited_sources:
            try:
                rows = _coll.get(
                    where={"heading": cs["heading"]},
                    include=["metadatas"],
                    limit=1,
                )
                if rows["metadatas"]:
                    m = rows["metadatas"][0]
                    cs["source_idx"] = m.get("source_idx", -1)
                    # source_idx_start may be missing in older indexed docs — fall back to source_idx
                    cs["source_idx_start"] = m.get("source_idx_start", cs["source_idx"])
            except Exception:
                cs.setdefault("source_idx", -1)
                cs.setdefault("source_idx_start", -1)
    except Exception:
        pass

    sources = []
    for s in cited_sources:
        raw_page = s["pages"][0] if s.get("pages") else 0
        page = raw_page + 1  # MinerU is 0-indexed, viewer is 1-indexed
        sources.append(
            {
                "doc_id": target_id,
                "page": page,
                "label": f"p.{page} — {s['heading'][:40]}",
                "heading": s["heading"],
                "source_idx_start": s.get("source_idx_start", s.get("source_idx", -1)),
                "source_idx": s.get("source_idx", -1),
                "pages_0idx": s.get("pages", [0]),
            }
        )
    return sources


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    target_id, err = _resolve_target(req.doc_ids, req.selected_doc_id)

    async def generator():
        if err:
            yield _sse({"type": "error", "message": err})
            return

        entry = doc_registry[target_id]
        doc_name = entry["name"]
        coll_name = entry.get("collection_name") or collection_name_for(doc_name)

        # ── Try real agent path first ─────────────────────────────────────────
        try:
            from langchain_core.tools import tool as lc_tool
            from langchain_openai import ChatOpenAI
            from streaming import stream_agent_turn

            # Build a doc-scoped answer_from_documents tool for this request
            @lc_tool
            def answer_from_documents(query: str) -> str:
                """Retrieve a grounded answer from the selected document."""
                result = answer_query(query, collection_name=coll_name)
                lines = [result["answer"], "", "Cited sources:"]
                for s in result["cited_sources"]:
                    lines.append(
                        f"  [Source {s['number']}] {s['heading']} | "
                        f"pages {s['pages']} | score {s['score']:.3f}"
                    )
                # Stash full result on the function for later source extraction
                answer_from_documents._last_result = result
                return "\n".join(lines)

            answer_from_documents._last_result = None

            # Try importing the agent factory — use deepagents if available,
            # fall back to langchain create_react_agent
            try:
                from deepagents import create_deep_agent
                from agent import SYSTEM_PROMPT as _SYSTEM_PROMPT

                llm = ChatOpenAI(model=settings.openai_model, temperature=0)
                agent = create_deep_agent(
                    tools=[answer_from_documents],
                    system_prompt=_SYSTEM_PROMPT,
                    model=llm,
                )
            except ImportError:
                from langchain.agents import create_react_agent, AgentExecutor
                from langchain import hub
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model=settings.openai_model, temperature=0)
                prompt = hub.pull("hwchase17/react")
                react_agent = create_react_agent(llm, [answer_from_documents], prompt)
                agent = AgentExecutor(
                    agent=react_agent, tools=[answer_from_documents], verbose=False
                )

            conversation = [{"role": "user", "content": req.question}]
            q: asyncio.Queue[dict | None] = asyncio.Queue()

            async def _run_agent():
                try:
                    reply = await stream_agent_turn(agent, conversation, event_queue=q)
                    # Extract sources from the last tool call result
                    last = answer_from_documents._last_result
                    cited = last["cited_sources"] if last else []
                    sources = _build_sources(cited, target_id, coll_name)
                    await q.put({"type": "answer", "answer": reply, "sources": sources})
                except Exception as exc:
                    await q.put({"type": "error", "message": str(exc)})
                finally:
                    await q.put(None)

            asyncio.create_task(_run_agent())

            while True:
                event = await q.get()
                if event is None:
                    break
                yield _sse(event)

        # ── Fallback: plain RAG with simulated progress ───────────────────────
        except ImportError:
            yield _sse({"type": "searching", "query": req.question})
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: answer_query(req.question, collection_name=coll_name),
                )
                yield _sse({"type": "retrieved"})
                sources = _build_sources(
                    result.get("cited_sources", []), target_id, coll_name
                )
                yield _sse(
                    {
                        "type": "answer",
                        "answer": result["answer"],
                        "sources": sources,
                    }
                )
            except Exception as exc:
                yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /ingest ───────────────────────────────────────────────────────────────────


@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Save file to data/docs/inputs/, kick off MinerU pipeline in background.
    Returns immediately with status='processing'.
    """
    allowed = {".pdf", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    doc_id = str(uuid.uuid4())
    original_name = file.filename or f"document_{doc_id}"

    # Prefix with doc_id so filenames never collide (same pattern as app.py)
    safe_stem = f"{doc_id}_{Path(original_name).stem}"
    dest = DOCS_INPUT_DIR / f"{safe_stem}{suffix}"

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    pages = _get_page_count(dest, suffix)

    doc_registry[doc_id] = {
        "name": original_name,
        "path": dest,
        "pages": pages,
        "collection_name": collection_name_for(original_name),
    }
    _set_status(doc_id, "processing", "Queued for indexing …")
    _save_registry()

    background_tasks.add_task(_run_pipeline, doc_id, dest)

    return {
        "doc_id": doc_id,
        "name": original_name,
        "pages": pages,
        "collection_name": collection_name_for(original_name),
        "status": "processing",
    }


# ── /status/{doc_id} ──────────────────────────────────────────────────────────


@app.get("/status/{doc_id}")
async def get_status(doc_id: str):
    if doc_id not in job_status:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    s = job_status[doc_id]
    return {
        "doc_id": doc_id,
        "status": s["status"],
        "message": s["message"],
        "pages": doc_registry.get(doc_id, {}).get("pages", 1),
    }


# ── /docs-list ────────────────────────────────────────────────────────────────


@app.get("/docs-list")
async def list_docs():
    """
    Merges in-memory jobs with kb_setup doc_registry entries so the UI can
    fully restore the document list after a page refresh.
    """
    seen_names: set[str] = set()
    result = []

    for doc_id, entry in doc_registry.items():
        js = job_status.get(doc_id, {"status": "unknown", "message": ""})
        result.append(
            {
                "doc_id": doc_id,
                "name": entry["name"],
                "pages": entry.get("pages", 1),
                "status": js["status"],
                "message": js.get("message", ""),
            }
        )
        seen_names.add(entry["name"])

    # Pull in anything from kb_setup doc_registry not already represented
    import chromadb as _chromadb

    for entry in list_all():
        if entry["filename"] in seen_names:
            continue
        coll_name = entry["collection_name"]
        try:
            _chromadb.PersistentClient(path=settings.chroma_db_path).get_collection(
                coll_name
            )
        except Exception:
            continue  # stale entry — collection gone

        synthetic_id = f"reg_{coll_name}"
        if synthetic_id in doc_registry:
            continue

        doc_registry[synthetic_id] = {
            "name": entry["filename"],
            "path": Path(""),
            "pages": entry.get("chunk_count", 1),
            "collection_name": coll_name,
        }
        job_status[synthetic_id] = {
            "status": "indexed",
            "message": "Restored from kb_setup registry.",
        }
        result.append(
            {
                "doc_id": synthetic_id,
                "name": entry["filename"],
                "pages": entry.get("chunk_count", 1),
                "status": "indexed",
                "message": "Restored from kb_setup registry.",
            }
        )

    return result


# ── /file/{doc_id} ────────────────────────────────────────────────────────────


@app.get("/file/{doc_id}")
async def get_file(doc_id: str):
    entry = doc_registry.get(doc_id)
    if not entry:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    path = Path(entry["path"])
    if not path.exists():
        raise HTTPException(404, f"File not on disk: {path.name}")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{entry["name"]}"'},
    )


# ── /health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_loaded": len(doc_registry),
        "docs_indexed": sum(1 for s in job_status.values() if s["status"] == "indexed"),
    }


# ── /bbox/{doc_id} ────────────────────────────────────────────────────────────
# Returns bounding boxes for every content_list item in [source_idx_start, source_idx_end].
# This covers the full chunk — heading, all paragraphs, tables, images.
#
# Query params:
#   start  — first source_idx in the chunk range (inclusive)
#   end    — last source_idx in the chunk range (inclusive)
#
# Returns:
#   [ { source_idx, page_idx, bbox: [x0,y0,x1,y1], type }, ... ]
#   bbox is in PDF points — same coordinate space as pdf.js viewport transform.


@app.get("/bbox/{doc_id}")
async def get_bbox(
    doc_id: str,
    start: int = Query(..., description="source_idx_start of the chunk"),
    end: int = Query(..., description="source_idx (end) of the chunk"),
):
    entry = doc_registry.get(doc_id)
    if not entry:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")

    content_list = _load_content_list(doc_id, entry)
    if content_list is None:
        raise HTTPException(404, "content_list.json not found for this document")

    if start < 0 or end < start:
        raise HTTPException(400, f"Invalid range: start={start} end={end}")

    # Compute MinerU's page dimensions by scanning ALL items on the same page(s)
    # as the requested chunk. Max x1/y1 across all items = MinerU's page size.
    target_pages: set[int] = set()
    for idx in range(start, min(end + 1, len(content_list))):
        p = content_list[idx].get("page_idx", 0)
        target_pages.add(p)

    page_sizes: dict[int, dict] = {}
    for item in content_list:
        p = item.get("page_idx", 0)
        if p not in target_pages:
            continue
        bbox = _extract_bbox(item)
        if bbox is None:
            continue
        if p not in page_sizes:
            page_sizes[p] = {"w": 0.0, "h": 0.0}
        page_sizes[p]["w"] = max(page_sizes[p]["w"], bbox[2])
        page_sizes[p]["h"] = max(page_sizes[p]["h"], bbox[3])

    results = []
    for idx in range(start, min(end + 1, len(content_list))):
        item = content_list[idx]
        bbox = _extract_bbox(item)
        if bbox is None:
            continue
        p = item.get("page_idx", 0)
        results.append(
            {
                "source_idx": idx,
                "page_idx": p,
                "bbox": bbox,
                "type": item.get("type", "text"),
                # MinerU coordinate space size for this page
                "mineru_page_w": page_sizes.get(p, {}).get("w", 0),
                "mineru_page_h": page_sizes.get(p, {}).get("h", 0),
            }
        )

    return results


def _extract_bbox(item: dict) -> list[float] | None:
    """Extract [x0, y0, x1, y1] from a MinerU content_list item."""
    bbox = item.get("bbox") or item.get("bounding_box")
    if bbox and len(bbox) == 4:
        return [float(v) for v in bbox]
    # Some MinerU versions use a polygon: [x0,y0, x1,y0, x1,y1, x0,y1]
    poly = item.get("poly")
    if poly and len(poly) >= 4:
        xs = [float(poly[i]) for i in range(0, len(poly), 2)]
        ys = [float(poly[i]) for i in range(1, len(poly), 2)]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _load_content_list(doc_id: str, entry: dict) -> list | None:
    """
    Find and load the _content_list.json for a document.
    Looks in the MinerU output directory using the original filename stem.
    """
    original_name = entry.get("name", "")
    file_path = Path(entry.get("path", ""))

    # The file saved to inputs is named  <doc_id>_<original_stem><suffix>
    # MinerU output dir is keyed on that same full stem
    stem = file_path.stem  # e.g. "abc123_myreport"

    candidate = DOCS_OUTPUT_DIR / stem / "auto" / f"{stem}_content_list.json"
    if candidate.exists():
        with open(candidate, encoding="utf-8") as f:
            return json.load(f)

    # Fallback: search by original name stem (for docs indexed before this server)
    orig_stem = Path(original_name).stem
    for path in DOCS_OUTPUT_DIR.rglob(f"{orig_stem}_content_list.json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return None


# ── Static files — serve index.html  (MUST be mounted last) ──────────────────
_index_html = ROOT / "index.html"
if _index_html.exists():
    app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


# ── Page-count helper ─────────────────────────────────────────────────────────


def _get_page_count(path: Path, suffix: str) -> int:
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(path)).pages)
        except Exception:
            pass
    return 1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
