# app/memory_retrieve.py
from typing import List, Dict, Tuple, Optional, Callable
import logging, math, os, json
from google.cloud import bigquery

BQ_DATASET = "neuromem"
BQ_TABLE   = "chunks"
DEFAULT_POOL = 200

# ---- toggle: local (no BQ) vs BigQuery ----
NM_USE_BQ = os.environ.get("NM_USE_BQ", "1") == "1"
LOCAL_CHUNKS_PATH = os.environ.get("NM_LOCAL_CHUNKS", "telemetry/local_chunks.jsonl")

# ---- Local embed helper ----
try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _ST_MODEL = None

def local_embed(text: str) -> List[float]:
    if _ST_MODEL is None:
        raise RuntimeError("Local embedder not available. Run: pip install sentence-transformers")
    vec = _ST_MODEL.encode([text], convert_to_numpy=True)[0]
    return [float(x) for x in vec.tolist()]

# ---- Basic math helpers ----
def _cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0: return 0.0
    dot = sa = sb = 0.0
    for i in range(n):
        ai = float(a[i]); bi = float(b[i])
        dot += ai * bi; sa += ai * ai; sb += bi * bi
    return 0.0 if sa == 0.0 or sb == 0.0 else dot / (math.sqrt(sa) * math.sqrt(sb))

# ---- Local provider (no GCP usage) ----
def _iter_local_chunks():
    if os.path.exists(LOCAL_CHUNKS_PATH):
        with open(LOCAL_CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        # fallback: tiny built-in set
        yield {"chunk_id": "doc_fallback_0", "doc_id":"doc_fallback",
               "text": "Local mode is active. Provide telemetry/local_chunks.jsonl for your own memory.",
               "pagerank": 0.0}

def _retrieve_local(query_text: str, embed_fn, alpha: float, k: int) -> Tuple[List[Dict], Dict]:
    q_vec = embed_fn(query_text)
    cands = []
    # collect first to compute PR normalization
    rows = list(_iter_local_chunks())
    pr_vals = [float(r.get("pagerank", 0.0)) for r in rows] or [0.0]
    min_pr, max_pr = min(pr_vals), max(pr_vals)
    denom = (max_pr - min_pr) or 1.0
    for r in rows:
        cosine = _cosine(q_vec, embed_fn(r.get("text","")))
        pr = float(r.get("pagerank", 0.0))
        pr_norm = (pr - min_pr) / denom if denom else 0.0
        blend = float(alpha * cosine + (1.0 - alpha) * pr_norm)
        cands.append({
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text"),
            "pagerank": pr,
            "cosine": cosine,
            "blend": blend
        })
    top = sorted(cands, key=lambda x: x["blend"], reverse=True)[:k]
    return top, {"alpha": alpha, "k": k, "pool": 0, "method": "local"}

# ---- BigQuery provider (read-only) ----
def _get_client(project: Optional[str] = None) -> bigquery.Client:
    return bigquery.Client(project=project) if project else bigquery.Client()

def _detect_vector_column_type(client: bigquery.Client) -> str:
    try:
        table = client.get_table(f"{client.project}.{BQ_DATASET}.{BQ_TABLE}")
        for f in table.schema:
            if f.name == "vector":
                t = (getattr(f, "field_type", "") or "").upper()
                if t == "VECTOR": return "VECTOR"
                if f.mode == "REPEATED" and t in ("FLOAT","FLOAT64"): return "ARRAY"
        return "ARRAY"
    except Exception as e:
        logging.warning("detect vector type failed: %s", e)
        return "ARRAY"

def _retrieve_bq(query_text: str, embed_fn, alpha: float, k: int, pool: int) -> Tuple[List[Dict], Dict]:
    client = _get_client()
    q_vec = embed_fn(query_text)
    vector_type = _detect_vector_column_type(client)
    method_used = "unknown"
    candidates: List[Dict] = []

    if vector_type == "VECTOR":
        try:
            sql = f"""
            DECLARE qvec VECTOR<FLOAT32> DEFAULT @qvec;
            WITH cands AS (
              SELECT chunk_id, text, pagerank,
                     VECTOR_COSINE_SIMILARITY(vector, qvec) AS cosine
              FROM {client.project}.{BQ_DATASET}.{BQ_TABLE}
              WHERE vector IS NOT NULL
              ORDER BY cosine DESC
              LIMIT @pool
            )
            SELECT chunk_id, text, pagerank, cosine FROM cands;
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("qvec", "VECTOR<FLOAT32>", q_vec),
                    bigquery.ScalarQueryParameter("pool", "INT64", pool),
                ]
            )
            rows = client.query(sql, job_config=job_config).result()
            for r in rows:
                rd = dict(r)
                candidates.append({
                    "chunk_id": rd.get("chunk_id"),
                    "text": rd.get("text"),
                    "pagerank": float(rd.get("pagerank") or 0.0),
                    "cosine": float(rd.get("cosine") or 0.0),
                })
            method_used = "bq_vector_sql"
        except Exception as e:
            logging.warning("VECTOR SQL path failed, falling back to ARRAY/python: %s", e)
            vector_type = "ARRAY"

    if vector_type != "VECTOR":
        try:
            limit = max(pool * 5, 500)
            sql = f"""
            SELECT chunk_id, text, pagerank, vector
            FROM {client.project}.{BQ_DATASET}.{BQ_TABLE}
            WHERE vector IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT @limit
            """
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            ])
            rows = client.query(sql, job_config=job_config).result()
            for r in rows:
                rd = dict(r)
                vec = rd.get("vector")
                if vec is None: continue
                try:
                    vlist = list(vec)
                except Exception:
                    vlist = vec
                cosine = _cosine(q_vec, vlist)
                candidates.append({
                    "chunk_id": rd.get("chunk_id"),
                    "text": rd.get("text"),
                    "pagerank": float(rd.get("pagerank") or 0.0),
                    "cosine": float(cosine),
                })
            method_used = "python_fallback"
        except Exception as e:
            logging.error("ARRAY/python fallback failed: %s", e)
            candidates = []

    if not candidates:
        return [], {"alpha": alpha, "k": k, "pool": pool, "method": method_used}

    pr_vals = [c.get("pagerank", 0.0) for c in candidates]
    min_pr, max_pr = min(pr_vals), max(pr_vals)
    denom = (max_pr - min_pr) or 1.0

    scored = []
    for c in candidates:
        pr_norm = (c.get("pagerank", 0.0) - min_pr) / denom if denom else 0.0
        cosine = c.get("cosine", 0.0)
        blend = float(alpha * cosine + (1.0 - alpha) * pr_norm)
        scored.append({
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text"),
            "pagerank": c.get("pagerank", 0.0),
            "cosine": cosine,
            "blend": blend
        })
    top = sorted(scored, key=lambda x: x["blend"], reverse=True)[:k]
    return top, {"alpha": alpha, "k": k, "pool": pool, "method": method_used}

# ---- Public API (chooses provider) ----
def retrieve_with_alpha(query_text: str,
                        embed_fn: Optional[Callable[[str], List[float]]] = None,
                        alpha: float = 0.5,
                        k: int = 5,
                        pool: int = DEFAULT_POOL,
                        project: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    if embed_fn is None:
        embed_fn = local_embed
    if not NM_USE_BQ:
        return _retrieve_local(query_text, embed_fn, alpha, k)
    return _retrieve_bq(query_text, embed_fn, alpha, k, pool)
