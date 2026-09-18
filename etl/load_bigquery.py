"""
Load raw and cleaned sales data into BigQuery.

Creates two tables:
  - raw_sales_transactions   (the untouched source data, for auditability/lineage)
  - sales_transactions_clean (the ETL output, used by dashboards + the ML model)

Setup:
  1. Create a GCP project + enable the BigQuery API
  2. Create a service account with 'BigQuery Data Editor' + 'BigQuery Job User' roles
  3. Download its JSON key and set the env var:
       export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
  4. pip install google-cloud-bigquery

Usage:
    python etl/load_bigquery.py \
        --project YOUR_GCP_PROJECT_ID \
        --dataset sales_analytics \
        --raw data/raw/Sales_transactions_2022_2025.csv \
        --clean data/processed/sales_clean.csv
"""
import argparse

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_dataset(client: bigquery.Client, project: str, dataset: str, location: str):
    dataset_id = f"{project}.{dataset}"
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_id} already exists")
    except NotFound:
        ds = bigquery.Dataset(dataset_id)
        ds.location = location
        client.create_dataset(ds)
        print(f"Created dataset {dataset_id}")


def load_table(client: bigquery.Client, df: pd.DataFrame, table_id: str):
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # wait for completion
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}")


def run(project: str, dataset: str, raw_path: str, clean_path: str, location: str):
    client = bigquery.Client(project=project)
    ensure_dataset(client, project, dataset, location)

    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(clean_path)

    load_table(client, raw_df, f"{project}.{dataset}.raw_sales_transactions")
    load_table(client, clean_df, f"{project}.{dataset}.sales_transactions_clean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load sales data into BigQuery")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", default="sales_analytics", help="BigQuery dataset name")
    parser.add_argument("--raw", default="data/raw/Sales_transactions_2022_2025.csv")
    parser.add_argument("--clean", default="data/processed/sales_clean.csv")
    parser.add_argument("--location", default="US", help="BigQuery dataset location")
    args = parser.parse_args()
    run(args.project, args.dataset, args.raw, args.clean, args.location)
