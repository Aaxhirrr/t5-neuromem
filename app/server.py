from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import time, uuid, os

from google.cloud import bigquery
from app.memory_retrieve import retrieve_with_alpha
from app.inference import generate_answer, build_prompt, count_tokens
from telemetry.logger import log_local  # new

app = FastAPI(title='T5-NeuroMem', version='0.1.0')
BQ_DATASET = 'neuromem'

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

@app.get('/')
def root():
    return {'status': 'ok', 'message': 'T5-NeuroMem server running 🚀'}

@app.get('/health')
def health():
    return {'ok': True}

def _row_dict(text: str, alpha: float, k: int, answer: str,
              citations: List[str], token_in: int, token_out: int, latency_ms: int) -> Dict:
    return {
        'query_id': str(uuid.uuid4()),
        'user_id': 'local',
        'text': text,
        'q_vector': None,   # optional
        'alpha': float(alpha),
        'k': int(k),
        'latency_ms': int(latency_ms),
        'response_text': answer,
        'citations': citations,
        'token_in': int(token_in),
        'token_out': int(token_out),
        'cost_usd': 0.0,
    }

def _log_query(text, alpha, k, answer, citations, token_in, token_out, latency_ms):
    sink = os.environ.get('LOG_SINK', 'local')
    row = _row_dict(text, alpha, k, answer, citations, token_in, token_out, latency_ms)
    if sink == 'bq_dml':
        # Requires billing enabled
        project = os.environ.get('GOOGLE_CLOUD_PROJECT') or None
        client = bigquery.Client(project=project) if project else bigquery.Client()
        table = f"{client.project}.{BQ_DATASET}.queries"
        sql = f'''
        INSERT INTO {table}
          (query_id, user_id, text, q_vector, alpha, k, latency_ms,
           response_text, citations, token_in, token_out, cost_usd)
        VALUES
          (@id, @uid, @text, NULL, @alpha, @k, @lat, @resp, @cits, @tin, @tout, @cost)
        '''
        params = [
            bigquery.ScalarQueryParameter('id','STRING', row['query_id']),
            bigquery.ScalarQueryParameter('uid','STRING', row['user_id']),
            bigquery.ScalarQueryParameter('text','STRING', row['text']),
            bigquery.ScalarQueryParameter('alpha','FLOAT64', row['alpha']),
            bigquery.ScalarQueryParameter('k','INT64', row['k']),
            bigquery.ScalarQueryParameter('lat','INT64', row['latency_ms']),
            bigquery.ScalarQueryParameter('resp','STRING', row['response_text']),
            bigquery.ArrayQueryParameter('cits','STRING', row['citations']),
            bigquery.ScalarQueryParameter('tin','INT64', row['token_in']),
            bigquery.ScalarQueryParameter('tout','INT64', row['token_out']),
            bigquery.ScalarQueryParameter('cost','FLOAT64', row['cost_usd']),
        ]
        bigquery.Client(project=project).query(
            sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
        ).result()
    else:
        # default: local NDJSON (free tier friendly)
        log_local(row)

@app.post('/predict', response_model=PredictResponse)
def predict(req: QueryRequest):
    t0 = time.perf_counter()
    res = retrieve_with_alpha(req.text, alpha=req.alpha, k=req.k)
    chunks, _meta = (res if (isinstance(res, (list, tuple)) and len(res) == 2 and isinstance(res[1], dict)) else (res, {}))
    answer = generate_answer(req.text, chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # tokens
    prompt = build_prompt(req.text, chunks)
    token_in = count_tokens(prompt)
    token_out = count_tokens(answer)
    citations = [c.get('chunk_id') for c in chunks]

    # telemetry (local by default)
    try:
        _log_query(req.text, req.alpha, req.k, answer, citations, token_in, token_out, latency_ms)
    except Exception as e:
        print('telemetry skipped:', e)

    return {
        'answer': answer,
        'alpha': req.alpha,
        'k': req.k,
        'citations': citations,
        'chunks': chunks,
    }
