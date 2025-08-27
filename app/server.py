from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import time, uuid, os, json
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.memory_retrieve import retrieve_with_alpha, LOCAL_CHUNKS_PATH
from app.inference import generate_answer, build_prompt, count_tokens
from telemetry.logger import log_local
from app.pagerank_local import recompute_pagerank

app = FastAPI(title="T5-NeuroMem", version="0.2.0")

# CORS for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    text: str
    alpha: float = 0.5
    k: int = 3

class PredictResponse(BaseModel):
    answer: str
    alpha: float
    k: int
    citations: List[str]
    chunks: List[Dict]

class IngestItem(BaseModel):
    text: str
    doc_id: Optional[str] = "local"

class IngestBatch(BaseModel):
    items: List[IngestItem]

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/mode")
def mode():
    return {
        "NM_USE_BQ": os.environ.get("NM_USE_BQ", "0"),
        "LOG_SINK": os.environ.get("LOG_SINK", "local"),
        "memory_file": LOCAL_CHUNKS_PATH
    }

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h3>T5-NeuroMem</h3><p>Try <a href='/demo'>/demo</a> or POST /predict</p>"

# Static demo
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/demo", response_class=HTMLResponse)
def demo():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

def _row_dict(text, alpha, k, answer, citations, token_in, token_out, latency_ms):
    return {
        "query_id": str(uuid.uuid4()), "user_id": "local", "text": text,
        "q_vector": None, "alpha": float(alpha), "k": int(k),
        "latency_ms": int(latency_ms), "response_text": answer,
        "citations": citations, "token_in": int(token_in),
        "token_out": int(token_out), "cost_usd": 0.0,
    }

def _log_query(text, alpha, k, answer, citations, token_in, token_out, latency_ms):
    log_local(_row_dict(text, alpha, k, answer, citations, token_in, token_out, latency_ms))

@app.post("/predict", response_model=PredictResponse)
def predict(req: QueryRequest):
    t0 = time.perf_counter()
    res = retrieve_with_alpha(req.text, alpha=req.alpha, k=req.k)
    chunks, _meta = (res if isinstance(res, (list,tuple)) and len(res)==2 and isinstance(res[1], dict) else (res, {}))
    answer = generate_answer(req.text, chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    prompt = build_prompt(req.text, chunks)
    token_in = count_tokens(prompt); token_out = count_tokens(answer)
    citations = [c.get("chunk_id") for c in chunks]
    try: _log_query(req.text, req.alpha, req.k, answer, citations, token_in, token_out, latency_ms)
    except Exception as e: print("telemetry skipped:", e)
    return {"answer": answer, "alpha": req.alpha, "k": req.k, "citations": citations, "chunks": chunks}

@app.post("/ingest")
def ingest(batch: IngestBatch):
    os.makedirs(os.path.dirname(LOCAL_CHUNKS_PATH) or ".", exist_ok=True)
    appended = 0
    with open(LOCAL_CHUNKS_PATH, "a", encoding="utf-8") as f:
        for it in batch.items:
            if not it.text or not it.text.strip():
                continue
            cid = f"{(it.doc_id or 'local')}_{uuid.uuid4().hex[:8]}"
            record = {"chunk_id": cid, "doc_id": it.doc_id or "local", "text": it.text.strip(), "pagerank": 0.0}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            appended += 1
    if appended == 0:
        raise HTTPException(status_code=400, detail="No valid items to ingest.")
    n = recompute_pagerank()  # refresh PR after ingest
    return {"ok": True, "ingested": appended, "total_chunks": n}
