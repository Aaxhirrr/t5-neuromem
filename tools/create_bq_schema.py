# tools/create_bq_schema.py
from google.cloud import bigquery
import sys

PROJECT = None  # None -> uses default from ADC (gcloud config)
DATASET = "neuromem"
LOCATION = "US"

client = bigquery.Client(project=PROJECT)

project = client.project
dataset_id = f"{project}.{DATASET}"

print(f"Using project: {project}")
print(f"Creating dataset: {dataset_id} (location={LOCATION})")

# create dataset if missing
ds = bigquery.Dataset(dataset_id)
ds.location = LOCATION
try:
    client.create_dataset(ds, exists_ok=True)
    print("Dataset ready.")
except Exception as e:
    print("Failed creating dataset:", e)
    sys.exit(1)

# DDL for core tables (ARRAY vector fallback)
ddl_chunks = f"""
CREATE TABLE IF NOT EXISTS `{dataset_id}.chunks` (
  chunk_id STRING NOT NULL,
  doc_id STRING,
  text STRING,
  vector ARRAY<FLOAT64>,    -- fallback vector storage; change to VECTOR<FLOAT32> if your project supports it
  pagerank FLOAT64 DEFAULT 0.0,
  retention_score FLOAT64 DEFAULT 0.5,
  usage_count INT64 DEFAULT 0,
  last_used TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  links ARRAY<STRING>,
  tags ARRAY<STRING>
);
"""

ddl_queries = f"""
CREATE TABLE IF NOT EXISTS `{dataset_id}.queries` (
  query_id STRING NOT NULL,
  user_id STRING,
  text STRING,
  q_vector ARRAY<FLOAT64>,
  alpha FLOAT64,
  k INT64,
  latency_ms INT64,
  response_text STRING,
  citations ARRAY<STRING>,
  token_in INT64,
  token_out INT64,
  cost_usd FLOAT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
"""

ddl_evals = f"""
CREATE TABLE IF NOT EXISTS `{dataset_id}.evals` (
  eval_id STRING NOT NULL,
  dataset STRING,
  example_id STRING,
  with_mem BOOL,
  alpha FLOAT64,
  k INT64,
  metric STRING,
  rouge_l FLOAT64,
  em FLOAT64,
  f1 FLOAT64,
  latency_ms INT64,
  cost_usd FLOAT64,
  notes STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
"""

for sql in (ddl_chunks, ddl_queries, ddl_evals):
    try:
        print("Running DDL...")
        job = client.query(sql)
        job.result()
        print("OK")
    except Exception as e:
        print("DDL failed:", e)
        sys.exit(1)

print("All tables created in dataset:", dataset_id)
print("You can view them in the BigQuery console.")
