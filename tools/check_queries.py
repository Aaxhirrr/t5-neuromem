import os
from google.cloud import bigquery

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "t5-neuromem"

client = bigquery.Client(project=PROJECT)
sql = """
SELECT
  query_id, text, alpha, k, latency_ms, token_in, token_out,
  ARRAY_LENGTH(citations) AS cites, created_at
FROM `t5-neuromem.neuromem.queries`
ORDER BY created_at DESC
LIMIT 5
"""
rows = client.query(sql).result()
print([dict(r) for r in rows])
