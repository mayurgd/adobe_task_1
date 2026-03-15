# Insight Agent

A local document Q&A system — upload PDFs, ask questions, get cited answers with highlighted source locations in the viewer.

---

## Requirements

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) (auto-installed by `start.sh` if missing)
- An OpenAI API key

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd insight-agent
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# Optional
OPENAI_MODEL=gpt-4o
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

### 3. Run the startup script

```bash
chmod +x start.sh
./start.sh
```

The script will:

1. Check for Python 3.12 (warns if higher, errors if lower)
2. Create a `.venv` if one doesn't exist
3. Install all dependencies from `requirements.txt`
4. Install MinerU via `uv pip install -U "mineru[all]"`
5. Start the FastAPI server on `http://0.0.0.0:8000`
6. Open `http://localhost:8000` in your browser automatically

> On subsequent runs the venv and dependency steps are skipped — the server starts immediately.

---

## Usage

### Uploading documents

- Click **Browse** or drag and drop a PDF into the left panel
- The document is parsed by MinerU and indexed into ChromaDB
- A status badge shows `Indexing…` → `Indexed` when ready

### Asking questions

- Once at least one document is indexed the chat input unlocks
- Type a question and press **Enter**
- The agent plans its steps, retrieves relevant chunks, and streams a cited answer

### Viewing sources

- Source tags appear below each answer (e.g. `📄 p.4 — Revenue Overview`)
- Clicking a tag opens that page in the PDF viewer and highlights the exact passage

---

## Project Structure

```
├── index.html              # Frontend (single-file UI)
├── server.py               # FastAPI backend — ingest, status, ask, bbox endpoints
├── streaming.py            # SSE streaming helper for agent events
├── start.sh                # One-command startup script
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
└── src/
    ├── agent.py            # CLI agent + system prompt
    ├── config.py           # Settings via pydantic-settings
    └── kb_setup/
        ├── answer_query.py # RAG pipeline (retrieve → generate → cite)
        ├── chunker.py      # MinerU content list → overlapping text chunks
        ├── doc_registry.py # Registry persistence (data/registry.json)
        ├── index_documents.py  # CLI indexer
        ├── indexer.py      # Embed chunks → ChromaDB
        ├── retriever.py    # Bi-encoder search + cross-encoder rerank
        └── text_utils.py   # HTML table parsing, text cleaning
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Upload a PDF — starts background indexing |
| `GET` | `/status/{doc_id}` | Poll indexing status |
| `GET` | `/docs-list` | List all documents (used for session restore) |
| `GET` | `/file/{doc_id}` | Serve the raw PDF to the viewer |
| `POST` | `/ask` | SSE stream — agent progress + final answer |
| `GET` | `/bbox/{doc_id}` | Bounding boxes for source highlighting |
| `GET` | `/health` | Server health check |

---

## Configuration

All settings are in `src/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o` | LLM for answer generation |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Bi-encoder for indexing |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `MINERU_OUTPUT_DIR` | `data/docs/outputs` | MinerU parse output location |
| `DEFAULT_TOP_K` | `5` | Chunks returned per query |

---

## Troubleshooting

**MinerU parse fails**
Ensure MinerU is installed correctly: `uv pip install -U "mineru[all]"`. The CLI binary must be on `PATH` as `mineru`.

**ChromaDB "RustBindingsAPI" error**
Only one `PersistentClient` instance should exist per process — the server manages this automatically. If running scripts directly, avoid creating multiple clients in parallel.

**Embedding model slow on first run**
`BAAI/bge-base-en-v1.5` and the reranker are downloaded from HuggingFace on first use and cached locally.

**API offline warning in the UI**
The frontend cannot reach `http://localhost:8000`. Make sure the server is running (`./start.sh`) and no firewall is blocking port 8000.