"""
app.py

Leadership Insight Agent — FastAPI backend.

Upload flow
-----------
POST /ingest
  1. Save file to docs/inputs/
  2. Fire background task (MinerU -> chunk -> embed -> ChromaDB)
  3. Return immediately with status="processing"

GET /status/{doc_id}
  Returns current indexing status: "processing" | "indexed" | "error"

POST /ask  (SSE streaming)
  Streams agent progress events in real time:
    data: {"type": "todos",     "todos": [...]}
    data: {"type": "searching", "query": "..."}
    data: {"type": "retrieved"}
    data: {"type": "answer",    "answer": "...", "sources": [...]}
    data: {"type": "error",     "message": "..."}
  Falls back to a simple vector-search + OpenAI answer when the agent is
  not configured (no TOOLS / deepagents available).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from kb_setup.chunker import build_chunks
from kb_setup.indexer import load_embedding_model

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Leadership Insight Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DOCS_INPUT_DIR = ROOT / "data" / "docs" / "inputs"
DOCS_OUTPUT_DIR = Path(settings.mineru_output_dir)

DOCS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Registry persistence path ──────────────────────────────────────────────────
REGISTRY_PATH = ROOT / "data" / "registry.json"

# ── Global singletons ──────────────────────────────────────────────────────────
import chromadb

_chroma_client: chromadb.PersistentClient | None = None
_chroma_collection = None
_embed_model = None

COLLECTION_NAME = settings.collection_name


def _get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    _chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    try:
        _chroma_collection = _chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        _chroma_collection = _chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ── Registry persistence helpers ───────────────────────────────────────────────


def _load_registry() -> None:
    global doc_registry, job_status
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
            job_status[doc_id] = raw_status.get(
                doc_id, {"status": "indexed", "message": "Restored from registry."}
            )
            if job_status[doc_id]["status"] == "processing":
                job_status[doc_id] = {
                    "status": "error",
                    "message": "Server restarted during indexing — please re-upload.",
                }
        print(f"[registry] Restored {len(doc_registry)} document(s) from disk.")
    except Exception as exc:
        print(f"[registry] Failed to load registry: {exc}")


def _save_registry() -> None:
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        serialisable_registry = {
            doc_id: {**entry, "path": str(entry["path"])}
            for doc_id, entry in doc_registry.items()
        }
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"doc_registry": serialisable_registry, "job_status": job_status},
                f,
                indent=2,
            )
    except Exception as exc:
        print(f"[registry] Failed to save registry: {exc}")


@app.on_event("startup")
async def _startup():
    global _embed_model
    print("[startup] Loading doc registry from disk …")
    _load_registry()
    print("[startup] Loading embedding model …")
    _embed_model = load_embedding_model()
    print("[startup] Opening ChromaDB collection …")
    _get_collection()
    print("[startup] Ready.")


# ── In-memory registries ───────────────────────────────────────────────────────
doc_registry: dict[str, dict] = {}
job_status: dict[str, dict] = {}


# ── Schemas ────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    doc_id: str
    name: str
    pages: int
    path: str
    status: str


class StatusResponse(BaseModel):
    doc_id: str
    status: str
    message: str


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str]


class SourceRef(BaseModel):
    doc_id: str
    page: int
    label: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


# ── Status helper ──────────────────────────────────────────────────────────────


def _set_status(doc_id: str, status: str, message: str) -> None:
    job_status[doc_id] = {"status": status, "message": message}
    print(f"[pipeline] [{status.upper()}] {doc_id[:8]} — {message}")
    if status in ("indexed", "error"):
        _save_registry()


# ── Background pipeline ────────────────────────────────────────────────────────


def _run_pipeline(doc_id: str, file_path: Path) -> None:
    try:
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

        saved_stem = file_path.stem
        content_list_path = (
            DOCS_OUTPUT_DIR / saved_stem / "auto" / f"{saved_stem}_content_list.json"
        )
        if not content_list_path.exists():
            raise FileNotFoundError(f"MinerU output not found: {content_list_path}")

        _set_status(doc_id, "processing", "Chunking document …")
        with open(content_list_path, encoding="utf-8") as f:
            content = json.load(f)

        chunks = build_chunks(content)
        if not chunks:
            raise ValueError(
                "No chunks produced — document may be empty or unreadable."
            )

        for c in chunks:
            c["doc_id"] = doc_id

        _set_status(doc_id, "processing", f"Embedding {len(chunks)} chunks …")
        embed_texts = [c["embed_text"] for c in chunks]
        embeddings = _embed_model.encode(
            embed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=settings.embed_batch_size,
        ).tolist()

        ids = [c["id"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "type": c["type"],
                "heading": c["heading"],
                "text": c["text"],  # full text, no truncation
                "source_idx": c.get("source_idx", -1),
                "img_path": c.get("img_path", ""),
                "caption": c.get("caption", ""),
                "pages": json.dumps(c["pages"]),
            }
            for c in chunks
        ]

        collection = _get_collection()
        batch_size = settings.index_batch_size
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=embed_texts[i:end],
            )
            _set_status(doc_id, "processing", f"Storing chunks … {end}/{len(chunks)}")

        all_pages = {p for c in chunks for p in c.get("pages", [])}
        if doc_id in doc_registry and all_pages:
            doc_registry[doc_id]["pages"] = max(all_pages) + 1

        _set_status(doc_id, "indexed", f"Indexed {len(chunks)} chunks successfully.")

    except subprocess.TimeoutExpired:
        _set_status(doc_id, "error", "MinerU timed out after 10 minutes.")
    except Exception as exc:
        _set_status(doc_id, "error", str(exc)[:500])


# ── /ask  — SSE streaming ──────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/ask")
async def ask(body: AskRequest):
    not_ready = [
        d for d in body.doc_ids if job_status.get(d, {}).get("status") != "indexed"
    ]
    if not_ready:
        raise HTTPException(
            status_code=400,
            detail=f"Docs not yet indexed: {not_ready}",
        )
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    async def event_stream():
        q: asyncio.Queue[dict | None] = asyncio.Queue()

        try:
            from agent import SYSTEM_PROMPT
            from deepagents import create_deep_agent
            from langchain_openai import ChatOpenAI
            from tools import TOOLS
            from streaming import stream_agent_turn

            llm = ChatOpenAI(model=settings.openai_model, temperature=0)
            agent = create_deep_agent(
                tools=TOOLS,
                system_prompt=SYSTEM_PROMPT,
                model=llm,
            )

            conversation = [{"role": "user", "content": body.question}]

            async def _run_agent():
                try:
                    reply = await stream_agent_turn(agent, conversation, event_queue=q)
                    sources = await asyncio.to_thread(_get_sources, body)
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

        except ImportError:
            yield _sse({"type": "searching", "query": body.question})
            try:
                answer_data = await _plain_rag(body)
                yield _sse({"type": "retrieved"})
                yield _sse(
                    {
                        "type": "answer",
                        "answer": answer_data["answer"],
                        "sources": answer_data["sources"],
                    }
                )
            except Exception as exc:
                yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Plain RAG helper (fallback when agent not available) ──────────────────────


async def _plain_rag(body: AskRequest) -> dict:
    q_embedding = _embed_model.encode(body.question, normalize_embeddings=True).tolist()

    collection = _get_collection()
    where_filter = (
        {"doc_id": {"$in": body.doc_ids}}
        if len(body.doc_ids) > 1
        else {"doc_id": body.doc_ids[0]}
    )
    top_k = min(settings.default_top_k, collection.count() or 1)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    if not metadatas:
        return {
            "answer": "No relevant content found in the selected documents.",
            "sources": [],
        }

    context_parts = []
    for i, (meta, doc_text) in enumerate(zip(metadatas, documents)):
        pages = json.loads(meta.get("pages", "[0]"))
        page_str = ", ".join(str(p + 1) for p in pages)
        # Full text from metadata; fall back to embed_text if somehow missing
        full_text = meta.get("text") or doc_text
        context_parts.append(
            f"[Chunk {i+1} | doc={meta['doc_id'][:8]} | page(s)={page_str}]\n"
            f"{full_text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    import openai

    openai.api_key = settings.openai_api_key
    response = await asyncio.to_thread(
        lambda: openai.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert document analyst. "
                        "Answer using ONLY the retrieved context. "
                        "Cite chunk numbers like [Chunk 1] when you use them. "
                        "If the answer isn't in the context, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {body.question}",
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
    )
    answer = response.choices[0].message.content.strip()
    sources = _build_source_refs(metadatas)
    return {"answer": answer, "sources": sources}


def _get_sources(body: AskRequest) -> list[dict]:
    try:
        q_emb = _embed_model.encode(body.question, normalize_embeddings=True).tolist()
        collection = _get_collection()
        where_filter = (
            {"doc_id": {"$in": body.doc_ids}}
            if len(body.doc_ids) > 1
            else {"doc_id": body.doc_ids[0]}
        )
        top_k = min(settings.default_top_k, collection.count() or 1)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where_filter,
            include=["metadatas"],
        )
        return _build_source_refs(results["metadatas"][0])
    except Exception:
        return []


def _build_source_refs(metadatas: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    refs = []
    for meta in metadatas:
        pages = json.loads(meta.get("pages", "[0]"))
        page = pages[0] + 1 if pages else 1
        doc_id = meta["doc_id"]
        key = (doc_id, page)
        if key in seen:
            continue
        seen.add(key)
        doc_name = doc_registry.get(doc_id, {}).get("name", doc_id[:8])
        refs.append({"doc_id": doc_id, "page": page, "label": f"{doc_name} p.{page}"})
    return refs


# ── Other endpoints ────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{file.filename}"
    dest = DOCS_INPUT_DIR / safe_name

    with dest.open("wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    pages = _get_page_count(dest, suffix)
    doc_registry[doc_id] = {"name": file.filename, "path": dest, "pages": pages}
    _set_status(doc_id, "processing", "Queued for indexing …")
    _save_registry()
    background_tasks.add_task(_run_pipeline, doc_id, dest)

    return IngestResponse(
        doc_id=doc_id,
        name=file.filename,
        pages=pages,
        path=str(dest.relative_to(ROOT)),
        status="processing",
    )


@app.get("/status/{doc_id}", response_model=StatusResponse)
async def get_status(doc_id: str):
    if doc_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    s = job_status[doc_id]
    return StatusResponse(doc_id=doc_id, status=s["status"], message=s["message"])


@app.get("/docs-list")
async def list_docs():
    return [
        {
            "doc_id": k,
            "name": v["name"],
            "pages": v["pages"],
            "status": job_status.get(k, {}).get("status", "unknown"),
            "message": job_status.get(k, {}).get("message", ""),
        }
        for k, v in doc_registry.items()
    ]


@app.get("/file/{doc_id}")
async def get_file(doc_id: str):
    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    entry = doc_registry[doc_id]
    path = Path(entry["path"])
    name = entry["name"]
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found on disk: {path.name}"
        )
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{name}"'},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_loaded": len(doc_registry),
        "docs_indexed": sum(1 for s in job_status.values() if s["status"] == "indexed"),
    }


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


def _get_page_count(path: Path, suffix: str) -> int:
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(path)).pages)
        except Exception:
            pass
    return 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
