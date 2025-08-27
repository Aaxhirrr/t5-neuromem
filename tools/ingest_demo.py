# tools/ingest_demo.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to sys.path


"""
Small demo ingester:
- builds 2-3 short demo docs
- chunks each doc by simple char-window (safe fallback)
- embeds chunks using local sentence-transformers (app.memory_retrieve.local_embed)
- inserts chunks into t5-neuromem.neuromem.chunks
"""
from google.cloud import bigquery
import uuid
import time
from datetime import datetime, timezone
from app.memory_retrieve import local_embed

client = bigquery.Client(project="t5-neuromem")
dataset = f"{client.project}.neuromem"
table_id = f"{dataset}.chunks"

# Demo documents (replace or extend)
DOCS = {
    "doc1": (
        "T5 is an encoder-decoder Transformer model from Google that frames every NLP task as text-to-text. "
        "It's used for translation, summarization, QA, and more. The model can be fine-tuned for many tasks."
    ),
    "doc2": (
        "LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into attention and MLP layers, "
        "allowing parameter-efficient fine-tuning. Adapters are small and cheap to store and serve."
    ),
    "doc3": (
        "BigQuery supports large-scale analytics and can store vectors as arrays or VECTOR<FLOAT32> (if enabled). "
        "We use it as the MemoryBank for T5-NeuroMem."
    ),
}

def chunk_text_simple(text: str, max_chars: int = 400):
    """Naive chunker: split by sentences/paragraphs if possible, else sliding window."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        parts = [text]
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            # sliding window on whitespace to avoid cutting words aggressively
            start = 0
            while start < len(p):
                end = start + max_chars
                if end >= len(p):
                    out.append(p[start:].strip())
                    break
                # backtrack to last space
                split_at = p.rfind(" ", start, end)
                if split_at <= start:
                    split_at = end
                out.append(p[start:split_at].strip())
                start = split_at
    return out

def build_rows():
    rows = []
    for doc_id, text in DOCS.items():
        chunks = chunk_text_simple(text, max_chars=400)
        for i, chunk in enumerate(chunks):
            try:
                vec = local_embed(chunk)  # returns list[float]
            except Exception as e:
                raise RuntimeError("local_embed failed — make sure sentence-transformers is installed") from e
            row = {
                "chunk_id": f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}",
                "doc_id": doc_id,
                "text": chunk,
                "vector": [float(x) for x in vec],  # ARRAY<FLOAT64> column
                "pagerank": 0.0,
                "retention_score": 0.7,
                "usage_count": 0,
                # last_used/created_at/updated_at will be auto-handled by DDL defaults if omitted
            }
            rows.append(row)
    return rows

def insert_rows(rows):
    print("Inserting", len(rows), "rows into", table_id)
    errors = client.insert_rows_json(table_id, rows)  # simple JSON insert
    if errors:
        print("Insert returned errors:")
        print(errors)
    else:
        print("Insert succeeded.")
    return errors

if __name__ == "__main__":
    rows = build_rows()
    insert_rows(rows)
