from google.cloud import bigquery
import os
print('ADC file:', os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or 'using gcloud ADC')
try:
    bq = bigquery.Client()
    print('BigQuery project:', bq.project)
    datasets = list(bq.list_datasets(max_results=5))
    print('Sample datasets:', [d.dataset_id for d in datasets] if datasets else 'none found or no perms')
except Exception as e:
    print('ERROR:', type(e).__name__, e)
