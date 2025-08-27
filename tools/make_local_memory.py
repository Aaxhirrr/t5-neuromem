# tools/make_local_memory.py
import os, json, uuid

os.makedirs("telemetry", exist_ok=True)
path = "telemetry/local_chunks.jsonl"

DOCS = {
  "doc1": "T5 is an encoder-decoder Transformer model from Google that frames every NLP task as text-to-text. It's used for translation, summarization, QA, and more.",
  "doc2": "LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into attention and MLP layers, allowing parameter-efficient fine-tuning. Adapters are small and cheap to store and serve.",
  "doc3": "This demo uses a local memory file to avoid any BigQuery reads/writes. Retrieval runs entirely on your machine."
}

with open(path, "w", encoding="utf-8") as f:
  for did, text in DOCS.items():
    f.write(json.dumps({
      "chunk_id": f"{did}_0_{uuid.uuid4().hex[:8]}",
      "doc_id": did,
      "text": text,
      "pagerank": 0.0
    }) + "\n")

print("Wrote", path)
