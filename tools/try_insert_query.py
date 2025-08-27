import os, uuid
from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "t5-neuromem"
client = bigquery.Client(project=PROJECT)
table = f"{PROJECT}.neuromem.queries"

sql = f"""
INSERT INTO `{table}`
  (query_id, user_id, text, q_vector, alpha, k, latency_ms,
   response_text, citations, token_in, token_out, cost_usd)
VALUES
  (@id, 'local', 'TEST row', NULL, 0.5, 3, 123,
   'ok', ['test_chunk'], 10, 5, 0.0)
"""
job = client.query(sql, job_config=bigquery.QueryJobConfig(
    query_parameters=[bigquery.ScalarQueryParameter("id","STRING", str(uuid.uuid4()))]
))
job.result()
print("Inserted test row:", job.job_id)
