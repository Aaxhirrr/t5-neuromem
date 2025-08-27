# tools/ingest_demo_batch.py
import os, sys, json, uuid
from google.cloud import bigquery
from app.memory_retrieve import local_embed

# ensure project root importability if launched from tools/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "t5-neuromem"
client = bigquery.Client(project=PROJECT)

dataset = f"{client.project}.neuromem"
table_id = f"{dataset}.chunks"

# Demo documents (same as before)
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
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        parts = [text]
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + max_chars
                if end >= len(p):
                    out.append(p[start:].strip())
                    break
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
            vec = local_embed(chunk)
            row = {
                "chunk_id": f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}",
                "doc_id": doc_id,
                "text": chunk,
                "vector": [float(x) for x in vec],
                "pagerank": 0.0,
                "retention_score": 0.7,
                "usage_count": 0,
            }
            rows.append(row)
    return rows

def write_ndjson(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_ndjson_to_bq(ndjson_path):
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ignore_unknown_values=True,
    )
    with open(ndjson_path, "rb") as fh:
        load_job = client.load_table_from_file(fh, table_id, job_config=job_config)
    load_job.result()  # wait
    print("Loaded rows via batch load. Job id:", load_job.job_id)
    # optional: show table row count
    table = client.get_table(table_id)
    print("Table rows now:", table.num_rows)

if __name__ == "__main__":
    rows = build_rows()
    print("Built", len(rows), "rows")
    tmp_path = os.path.join(os.path.dirname(__file__), "tmp_ingest.ndjson")
    write_ndjson(rows, tmp_path)
    print("Wrote ndjson to", tmp_path)
    load_ndjson_to_bq(tmp_path)
    # cleanup if you want:
    # os.remove(tmp_path)
