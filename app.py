from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid
from pathlib import Path

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Leadership Insight Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs/inputs"
DOCS_DIR.mkdir(exist_ok=True)

# ── In-memory doc registry (replace with a DB later) ──────────────────────────
# { doc_id: { "name": str, "path": Path, "pages": int } }
doc_registry: dict[str, dict] = {}


# ── Schemas ────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    doc_id: str
    name: str
    pages: int
    path: str


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


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a PDF (or DOCX / TXT) upload, saves it to ./docs/<doc_id>_<filename>,
    and returns metadata.

    TODO:
    - Parse page count from the PDF (e.g. with pypdf or pdfplumber)
    - Chunk the document text
    - Embed chunks and upsert into a vector store
    - Persist doc_registry to a database
    """
    allowed = {".pdf", ".docx", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{file.filename}"
    dest = DOCS_DIR / safe_name

    # Save file to disk
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # TODO: replace with real page count extraction
    pages = _get_page_count(dest, suffix)

    doc_registry[doc_id] = {
        "name": file.filename,
        "path": dest,
        "pages": pages,
    }

    return IngestResponse(
        doc_id=doc_id,
        name=file.filename,
        pages=pages,
        path=str(dest.relative_to(ROOT)),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Accepts a natural-language question and a list of doc_ids to search over.
    Returns a grounded answer and source citations.

    TODO:
    - Embed body.question
    - Retrieve top-k chunks from the vector store filtered to body.doc_ids
    - Rerank chunks (optional)
    - Call LLM with retrieved context + question
    - Parse source references from LLM response
    """
    # Validate that all doc_ids are known
    unknown = [d for d in body.doc_ids if d not in doc_registry]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown doc_ids: {unknown}")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # ── Stub response ── replace entirely with RAG pipeline ──────────────────
    answer = (
        "RAG pipeline not yet implemented. "
        "Wire up your vector store retrieval and LLM call in the /ask endpoint."
    )
    sources: list[SourceRef] = []
    # ─────────────────────────────────────────────────────────────────────────

    return AskResponse(answer=answer, sources=sources)


@app.get("/docs-list")
async def list_docs():
    """Returns all currently registered documents."""
    return [
        {"doc_id": k, "name": v["name"], "pages": v["pages"]}
        for k, v in doc_registry.items()
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "docs_loaded": len(doc_registry)}


# Serve the frontend (index.html + assets) from the same directory
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_page_count(path: Path, suffix: str) -> int:
    """
    Best-effort page count. Returns 1 on failure.
    Install pypdf for PDF support:  pip install pypdf
    """
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
