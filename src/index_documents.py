"""
index_documents.py

Builds a ChromaDB vector store from MinerU _content_list.json output.

Two-stage retrieval strategy:
  Stage 1 — semantic search on embedded text
             - text chunks   : heading + paragraph text
             - table chunks  : heading + caption + table converted to plain text
             - image chunks  : heading + caption
  Stage 2 — linked asset fetch using stored metadata
             - tables : full HTML stored in metadata["table_html"]
             - images : file path stored in metadata["img_path"]

Usage:
    python index_documents.py
"""

import json
import re
import uuid
from io import StringIO
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTENT_LIST_PATH = Path(
    "/Users/mayurgd/Documents/CodingSpace/adobe_task_1/data/outputs/annual_reports/adbe-2023-annual-report/auto/"
    "/adbe-2023-annual-report_content_list.json"
)
# Images produced by MinerU sit next to the content_list JSON
IMAGES_BASE_DIR = CONTENT_LIST_PATH.parent / "images"

CHROMA_DB_PATH = "/Users/mayurgd/Documents/CodingSpace/adobe_task_1/data/chroma_db"
COLLECTION_NAME = "adbe_2023_annual_report"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Minimum characters a text chunk must have to be worth indexing
MIN_TEXT_LENGTH = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def table_html_to_text(html: str) -> str:
    """Convert an HTML table to a plain-text string via pandas."""
    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return ""
        return dfs[0].to_string(index=False)
    except Exception:
        return ""


def clean(text: str) -> str:
    """Collapse excessive whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 1: Build chunks
# ---------------------------------------------------------------------------


def build_chunks(content: list) -> list[dict]:
    """
    Walk the flat content list in reading order and produce chunks.

    Each chunk is a dict with:
        id          : unique string id
        embed_text  : text that will be embedded
        type        : "text" | "table" | "image"
        heading     : nearest heading above this chunk
        pages       : list of page indices (int)

        # type-specific payload stored as metadata
        text        : paragraph text (text chunks)
        source_idx  : index into original content_list (table / image chunks)
        img_path    : relative image path (image chunks, convenience copy)
        caption     : caption string (table / image chunks)
    """
    chunks = []

    current_heading = ""
    current_text_parts: list[str] = []
    current_pages: set[int] = set()

    def flush_text_chunk():
        nonlocal current_text_parts, current_pages
        if not current_text_parts:
            return
        body = "\n".join(current_text_parts)
        if len(body) < MIN_TEXT_LENGTH:
            current_text_parts = []
            current_pages = set()
            return
        embed_text = clean(f"{current_heading}\n{body}")
        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "embed_text": embed_text,
                "type": "text",
                "heading": current_heading,
                "text": body,
                "pages": sorted(current_pages),
                "source_idx": -1,
                "img_path": "",
                "caption": "",
            }
        )
        current_text_parts = []
        current_pages = set()

    for source_idx, item in enumerate(content):
        item_type = item.get("type")

        # ── text / heading ─────────────────────────────────────────────────
        if item_type == "text":
            text = clean(item.get("text", ""))
            if not text:
                continue

            if item.get("text_level"):  # heading
                flush_text_chunk()  # save previous section first
                current_heading = text
            else:  # paragraph
                current_text_parts.append(text)
                current_pages.add(item.get("page_idx", 0))

        # ── table ──────────────────────────────────────────────────────────
        elif item_type == "table":
            flush_text_chunk()  # flush pending text first

            html = item.get("table_body", "")
            caption_list = item.get("table_caption") or []
            caption = clean(caption_list[0]) if caption_list else ""

            # Convert table to plain text for embedding
            table_text = table_html_to_text(html)
            embed_text = clean(f"{current_heading}\n{caption}\n{table_text}")

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "embed_text": embed_text,
                    "type": "table",
                    "heading": current_heading,
                    "text": table_text,
                    "source_idx": source_idx,  # fetch full HTML from content_list at query time
                    "img_path": "",
                    "caption": caption,
                    "pages": [item.get("page_idx", 0)],
                }
            )

        # ── image ──────────────────────────────────────────────────────────
        elif item_type == "image":
            caption_list = item.get("image_caption") or []
            caption = clean(caption_list[0]) if caption_list else ""
            img_path = item.get("img_path", "")

            embed_text = clean(f"{current_heading}\n{caption}")

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "embed_text": embed_text,
                    "type": "image",
                    "heading": current_heading,
                    "text": caption,
                    "source_idx": source_idx,  # fetch full image record from content_list at query time
                    "img_path": img_path,  # convenience copy for quick access
                    "caption": caption,
                    "pages": [item.get("page_idx", 0)],
                }
            )

        # discarded / unknown types are ignored
        else:
            continue

    flush_text_chunk()  # flush final section

    return chunks


# ---------------------------------------------------------------------------
# Step 2: Embed + store in ChromaDB
# ---------------------------------------------------------------------------


def build_index(chunks: list[dict], model: SentenceTransformer):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Drop and recreate so we can re-run idempotently
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # ChromaDB default = cosine distance with HNSW (no extra config needed)
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} ...")
    embed_texts = [c["embed_text"] for c in chunks]
    embeddings = model.encode(
        embed_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    # ChromaDB metadata values must be str | int | float | bool
    # Lists (pages) are serialised to JSON strings and deserialised on retrieval
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "type": c["type"],
            "heading": c["heading"],
            "text": c["text"][:2000],  # plain-text preview; cap to keep metadata lean
            "source_idx": c.get(
                "source_idx", -1
            ),  # pointer back to content_list for stage 2
            "img_path": c["img_path"],
            "caption": c["caption"],
            "pages": json.dumps(c["pages"]),  # list → JSON string
        }
        for c in chunks
    ]
    documents = embed_texts  # Chroma also stores the raw document string

    # Upsert in batches of 500
    batch = 500
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=metadatas[i : i + batch],
            documents=documents[i : i + batch],
        )
        print(f"  Stored {min(i + batch, len(chunks))}/{len(chunks)} chunks")

    return collection


# ---------------------------------------------------------------------------
# Step 3: Two-stage retrieval (demo)
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    collection,
    model: SentenceTransformer,
    content: list,
    top_k: int = 5,
):
    """
    Stage 1 : semantic search → top-k chunks
    Stage 2 : hydrate tables/images by looking up source_idx in the original
              content_list — no bulky HTML stored in Chroma metadata.
    """
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Stage 2: fetch full asset from content_list using source_idx
        source_idx = meta.get("source_idx", -1)
        table_html = ""
        img_path = meta.get("img_path", "")
        if source_idx >= 0 and source_idx < len(content):
            raw = content[source_idx]
            if meta["type"] == "table":
                table_html = raw.get("table_body", "")
            elif meta["type"] == "image":
                img_path = raw.get("img_path", img_path)

        result = {
            "rank": i + 1,
            "score": round(1 - distance, 4),  # cosine similarity
            "type": meta["type"],
            "heading": meta["heading"],
            "pages": json.loads(meta["pages"]),
            "text": meta["text"],
            "table_html": table_html,  # fetched from source at query time
            "img_path": img_path,
            "caption": meta["caption"],
        }
        output.append(result)

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=== Loading content list ===")
    with open(CONTENT_LIST_PATH, encoding="utf-8") as f:
        content = json.load(f)
    print(f"Loaded {len(content)} raw elements")

    print("\n=== Building chunks ===")
    chunks = build_chunks(content)

    type_counts = {}
    for c in chunks:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"Total chunks : {len(chunks)}")
    for t, count in type_counts.items():
        print(f"  {t:8s}: {count}")

    print("\n=== Loading embedding model ===")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\n=== Building ChromaDB index ===")
    collection = build_index(chunks, model)
    print(f"\nIndex saved to: {CHROMA_DB_PATH}")
    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Total vectors : {collection.count()}")

    # Quick smoke test
    print("\n=== Smoke test retrieval ===")
    test_query = "What was Adobe's total revenue in fiscal 2023?"
    results = retrieve(test_query, collection, model, content, top_k=3)
    print(f"\nQuery: {test_query}\n")
    for r in results:
        print(
            f"[{r['rank']}] score={r['score']}  type={r['type']}  "
            f"pages={r['pages']}"
        )
        print(f"    heading : {r['heading'][:80]}")
        if r["type"] == "text":
            print(f"    text    : {r['text'][:200]}")
        elif r["type"] == "table":
            print(f"    caption : {r['caption']}")
            print(f"    html    : {r['table_html'][:120]}...")
        elif r["type"] == "image":
            print(f"    caption : {r['caption']}")
            print(f"    img_path: {r['img_path']}")
        print()


if __name__ == "__main__":
    main()
