from __future__ import annotations
from typing import List, Dict, Tuple
import json, os, math, uuid, tempfile, shutil

# Reuse the local embedder from memory_retrieve
from app.memory_retrieve import local_embed, LOCAL_CHUNKS_PATH

SIM_THRESHOLD = float(os.environ.get("NM_PR_SIM_THRESHOLD", "0.38"))
DAMPING = float(os.environ.get("NM_PR_DAMPING", "0.85"))
ITERS = int(os.environ.get("NM_PR_ITERS", "20"))

def _load_chunks(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path): return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0: return 0.0
    dot = sa = sb = 0.0
    for i in range(n):
        ai, bi = float(a[i]), float(b[i])
        dot += ai * bi; sa += ai*ai; sb += bi*bi
    return 0.0 if sa==0.0 or sb==0.0 else dot/(math.sqrt(sa)*math.sqrt(sb))

def _build_graph(rows: List[Dict]) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (ids, adj) where adj[u][v]=weight if sim>=threshold."""
    ids = [r.get("chunk_id") for r in rows]
    texts = [r.get("text","") for r in rows]
    vecs = [local_embed(t) for t in texts]

    adj: Dict[str, Dict[str, float]] = {cid:{} for cid in ids}
    n = len(ids)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            sim = _cosine(vecs[i], vecs[j])
            if sim >= SIM_THRESHOLD:
                adj[ids[i]][ids[j]] = sim
    return ids, adj

def _pagerank(ids: List[str], adj: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    n = len(ids)
    pr = {cid: 1.0/n for cid in ids}
    # precompute out-weight
    outw = {u: sum(adj[u].values()) for u in ids}
    # build reverse adjacency for efficiency
    incoming: Dict[str, List[Tuple[str,float]]] = {v: [] for v in ids}
    for u in ids:
        for v,w in adj[u].items():
            incoming[v].append((u,w))
    base = (1.0 - DAMPING) / n
    for _ in range(ITERS):
        new = {}
        for v in ids:
            s = 0.0
            for (u,w) in incoming[v]:
                denom = outw[u] or 1.0
                s += pr[u] * (w/denom)
            new[v] = base + DAMPING * s
        pr = new
    # min-max normalize for easier blend later (0..1)
    vals = list(pr.values()) or [0.0]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pr_norm = {k: (v - lo)/rng for k,v in pr.items()}
    return pr_norm

def recompute_pagerank(path: str = LOCAL_CHUNKS_PATH) -> int:
    rows = _load_chunks(path)
    if not rows: return 0
    ids, adj = _build_graph(rows)
    pr_norm = _pagerank(ids, adj)
    # write out with updated pagerank
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            r["pagerank"] = float(pr_norm.get(r.get("chunk_id"), 0.0))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.move(tmp, path)
    return len(rows)

if __name__ == "__main__":
    n = recompute_pagerank()
    print(f"Recomputed PageRank for {n} chunks -> {LOCAL_CHUNKS_PATH}")
