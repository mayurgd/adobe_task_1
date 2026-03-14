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

POST /ask
  Vector-search over the single ChromaDB collection filtered to the
  requested doc_ids, then call OpenAI for a grounded answer.
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
DOCS_OUTPUT_DIR = Path(settings.mineru_output_dir)  # from .env: MINERU_OUTPUT_DIR

DOCS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global singletons (loaded once at startup) ─────────────────────────────────
import chromadb  # noqa: E402

_chroma_client: chromadb.PersistentClient | None = None
_chroma_collection = None  # chromadb.Collection
_embed_model = None  # SentenceTransformer

COLLECTION_NAME = settings.collection_name


def _get_collection():
    """Return (or lazily create) the shared ChromaDB collection."""
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


@app.on_event("startup")
async def _startup():
    """Pre-load embedding model and open ChromaDB at server start."""
    global _embed_model
    print("[startup] Loading embedding model …")
    _embed_model = load_embedding_model()
    print("[startup] Opening ChromaDB collection …")
    _get_collection()
    print("[startup] Ready.")


# ── In-memory registries ───────────────────────────────────────────────────────
# doc_registry : doc_id -> { name, path, pages }
# job_status   : doc_id -> { status: "processing"|"indexed"|"error", message }

doc_registry: dict[str, dict] = {}
job_status: dict[str, dict] = {}


# ── Schemas ────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    doc_id: str
    name: str
    pages: int
    path: str
    status: str  # always "processing" on first response


class StatusResponse(BaseModel):
    doc_id: str
    status: str  # "processing" | "indexed" | "error"
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


# ── Background pipeline ────────────────────────────────────────────────────────


def _run_pipeline(doc_id: str, file_path: Path) -> None:
    """
    Blocking function executed in FastAPI's thread-pool:
      1. Run MinerU to parse the document.
      2. Locate the output _content_list.json.
      3. Chunk -> embed -> upsert into ChromaDB with doc_id metadata.
    """
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute hard cap per document
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU exited {result.returncode}.\n"
                f"stderr: {result.stderr[-800:]}"
            )

        # ── Step 2: Locate content_list.json ──────────────────────────────────
        # MinerU output convention:
        #   <DOCS_OUTPUT_DIR>/<stem>/auto/<stem>_content_list.json
        # The stem is derived from the *original* filename (without the doc_id prefix).
        saved_stem = (
            file_path.stem
        )  # uses the uuid-prefixed filename MinerU actually saw

        content_list_path = (
            DOCS_OUTPUT_DIR / saved_stem / "auto" / f"{saved_stem}_content_list.json"
        )
        if not content_list_path.exists():
            raise FileNotFoundError(f"MinerU output not found: {content_list_path}")

        # ── Step 3: Chunk ─────────────────────────────────────────────────────
        _set_status(doc_id, "processing", "Chunking document …")
        with open(content_list_path, encoding="utf-8") as f:
            content = json.load(f)

        chunks = build_chunks(content)
        if not chunks:
            raise ValueError(
                "No chunks produced — document may be empty or unreadable."
            )

        # Inject doc_id into every chunk for collection-level filtering
        for c in chunks:
            c["doc_id"] = doc_id

        # ── Step 4: Embed + upsert ────────────────────────────────────────────
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
                "text": c["text"][: settings.metadata_text_cap],
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

        # Update page count from actual MinerU output
        all_pages = {p for c in chunks for p in c.get("pages", [])}
        if doc_id in doc_registry and all_pages:
            doc_registry[doc_id]["pages"] = max(all_pages) + 1

        _set_status(doc_id, "indexed", f"Indexed {len(chunks)} chunks successfully.")

    except subprocess.TimeoutExpired:
        _set_status(doc_id, "error", "MinerU timed out after 10 minutes.")
    except Exception as exc:
        _set_status(doc_id, "error", str(exc)[:500])


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Save the uploaded file and kick off the background indexing pipeline.
    Returns immediately with status='processing'.
    """
    allowed = {".pdf", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{file.filename}"
    dest = DOCS_INPUT_DIR / safe_name

    with dest.open("wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    # Best-effort page count up front; updated after MinerU finishes
    pages = _get_page_count(dest, suffix)

    doc_registry[doc_id] = {
        "name": file.filename,
        "path": dest,
        "pages": pages,
    }
    _set_status(doc_id, "processing", "Queued for indexing …")

    # Add to FastAPI's background task queue (runs in thread pool)
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
    """Poll to track per-document indexing progress."""
    if doc_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    s = job_status[doc_id]
    return StatusResponse(
        doc_id=doc_id,
        status=s["status"],
        message=s["message"],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Embed the question, retrieve top-k chunks from ChromaDB filtered to
    body.doc_ids, then generate an answer via OpenAI.
    """
    # Guard: only query fully-indexed docs
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

    # Embed question
    q_embedding = _embed_model.encode(
        body.question,
        normalize_embeddings=True,
    ).tolist()

    # Retrieve from ChromaDB, filtered to selected docs
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
        return AskResponse(
            answer="No relevant content found in the selected documents.",
            sources=[],
        )

    # Build LLM context
    context_parts = []
    for i, (meta, doc_text) in enumerate(zip(metadatas, documents)):
        pages = json.loads(meta.get("pages", "[0]"))
        page_str = ", ".join(str(p + 1) for p in pages)  # 1-indexed
        context_parts.append(
            f"[Chunk {i+1} | doc={meta['doc_id'][:8]} | page(s)={page_str}]\n"
            f"{meta.get('text') or doc_text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Call OpenAI
    import openai  # lazy import

    openai.api_key = settings.openai_api_key

    response = await asyncio.to_thread(
        lambda: openai.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert document analyst. "
                        "Answer the user's question using ONLY the retrieved context. "
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

    # Build source refs
    sources: list[SourceRef] = []
    for meta in metadatas:
        pages = json.loads(meta.get("pages", "[0]"))
        page = pages[0] + 1 if pages else 1
        doc_name = doc_registry.get(meta["doc_id"], {}).get("name", meta["doc_id"][:8])
        sources.append(
            SourceRef(
                doc_id=meta["doc_id"],
                page=page,
                label=f"{doc_name} p.{page}",
            )
        )

    return AskResponse(answer=answer, sources=sources)


@app.get("/docs-list")
async def list_docs():
    """All registered documents with their current indexing status."""
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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs_loaded": len(doc_registry),
        "docs_indexed": sum(1 for s in job_status.values() if s["status"] == "indexed"),
    }


# Serve frontend from the same directory
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_page_count(path: Path, suffix: str) -> int:
    """Best-effort page count from a PDF. Returns 1 on failure."""
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(path)).pages)
        except Exception:
            pass
    return 1


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
