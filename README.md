# Sales Analytics Pipeline: ETL → BigQuery → PyTorch Forecasting → n8n

An end-to-end data pipeline built on a real (messy) 18,000-row retail transactions
dataset spanning 2022–2025, six countries, and four sales channels.

**Live demo:** [add your Streamlit Cloud URL here after deploying]
**Dashboard (BigQuery + Looker Studio):** [add your Looker Studio URL here]

## What this project demonstrates

- **ETL / data engineering**: profiling raw data, fixing real quality issues
  (inconsistent categorical casing, duplicate records, implausible values),
  and producing an analysis-ready dataset with documented before/after metrics.
- **Data warehousing (BigQuery)**: loading raw + cleaned tables, writing SQL views
  that answer real business questions (monthly trends, category/channel/geo
  performance).
- **Applied ML (PyTorch)**: an LSTM trained to forecast daily revenue, evaluated
  against a naive baseline — not just "here's an accuracy number."
- **Orchestration (n8n)**: a daily workflow with quality gates and alerting,
  not just a happy-path pipeline.
- **Honest, evidence-based decision-making**: see "A design decision worth
  reading" below — this is arguably the most important part of the project.

## Architecture

```
Sales_transactions_2022_2025.csv (raw, messy)
        │
        ▼
etl/clean.py  ──────────────►  data/processed/sales_clean.csv
   - standardizes casing            + data_quality_report.json
   - dedupes transactions
   - nulls implausible ages
   - derives analysis fields
        │
        ▼
etl/load_bigquery.py  ─────►  BigQuery: raw_sales_transactions
                                          sales_transactions_clean
        │
        ▼
sql/views.sql  ─────────────►  BigQuery views (monthly, category,
                                 channel, geo, daily revenue)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                                            ▼
        Looker Studio dashboard                    models/forecast.py (LSTM)
        (business-facing BI)                                    │
                                                                  ▼
                                                    app/dashboard.py (Streamlit)
                                                    - ETL quality report
                                                    - business overview
                                                    - interactive forecast

n8n/sales_pipeline_workflow.json orchestrates the whole thing daily:
  schedule → ETL → quality gate → BigQuery load → retrain forecast →
  model-quality gate → Slack notification
```

**Note on the n8n workflow:** it's built using **SSH nodes**, not the Execute
Command node. n8n is hosted on a Hostinger VPS inside Docker, and the
container itself doesn't have Python or this repo inside it — so each step
SSHes into the VPS host, where a Python venv with this repo is set up
separately. See the setup steps below.

## A design decision worth reading

The original plan was a **return-prediction model** (will this order be
returned?). Before building it, I checked whether the data actually supported
it: correlation analysis and a baseline logistic regression both showed
`Is_Return` is statistically unrelated to every other column in the dataset
(ROC-AUC ≈ 0.52, no better than a coin flip). The labels appear to be randomly
assigned in the source data — common in synthetic/generated datasets.

Rather than ship a model with no real predictive power, I re-profiled the data,
found genuine signal in the **time dimension** instead (consistent Nov/Dec
seasonality across all four years, ~2x January revenue every year), and
pivoted to **revenue forecasting**. The finished model beats a naive "same day
last week" baseline by ~28% (MAE $3,236 vs $4,474).

The negative result is left in the repo (`models/train.py`) rather than
deleted, because catching a modeling dead-end *before* shipping it is the
actual skill being demonstrated.

## Project structure

```
data/raw/                  Original CSV + data dictionary
data/processed/            Cleaned CSV, data quality report, forecast predictions
etl/clean.py                Data cleaning + feature engineering
etl/load_bigquery.py        Loads raw + clean tables into BigQuery
sql/views.sql                BigQuery views for BI
models/train.py             Return-prediction model (documented negative result)
models/forecast.py          Revenue forecasting LSTM (the model actually used)
models/artifacts/            Saved model weights, scaler, metrics
app/dashboard.py             Streamlit app (3 tabs: quality, BI, forecast)
n8n/sales_pipeline_workflow.json   Importable n8n orchestration workflow
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. Clean the raw data
python etl/clean.py

# 2. (Optional) Load to BigQuery — requires GOOGLE_APPLICATION_CREDENTIALS set
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
python etl/load_bigquery.py --project YOUR_GCP_PROJECT_ID

# 3. Train the forecasting model
python models/forecast.py

# 4. Run the dashboard
streamlit run app/dashboard.py
```

## Setting up the n8n workflow (Hostinger VPS)

1. SSH into your VPS, install Python: `apt update && apt install -y python3 python3-pip python3-venv`
2. Get this repo onto the VPS (`git clone`, or upload + unzip), e.g. to `/root/return-risk-pipeline`
3. `cd /root/return-risk-pipeline && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
4. Upload your GCP service account key to the VPS (e.g. `/root/gcp-key.json`) via SFTP — never commit it
5. In n8n, create an SSH credential pointed at your VPS (host, port 22, username, password or key)
6. Import `n8n/sales_pipeline_workflow.json`, then in each SSH node: set the credential you just created,
   and adjust the repo path / `GCP_PROJECT_ID` if yours differ from the defaults in the workflow
7. Test manually first: `Execute workflow` in n8n, or run the commands yourself over SSH, before
   relying on the daily schedule

## Data source

Retail sales transactions dataset (2022–2025), 18,045 rows, 36 columns,
covering the US, Australia, France, Canada, the UK, and Germany. See
`data/raw/sales_data_dictionary.csv` for the full field dictionary.

## Honest limitations

- The forecasting model is trained on ~4 years of daily data (1,461 points) —
  enough to capture yearly seasonality, but a longer history would give more
  confidence in the trend estimate.
- MAPE (48.7%) is high in absolute terms because daily revenue is genuinely
  noisy at this transaction volume; the model is evaluated against a baseline
  for this reason rather than on MAPE alone.
- `Return_Flag`, `Customer_Rating`, and order cancellations show no learnable
  relationship to other fields in this dataset — flagged explicitly rather
  than worked around.
