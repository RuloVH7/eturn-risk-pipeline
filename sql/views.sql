-- SQL views for BigQuery, built on top of `sales_transactions_clean`
-- (loaded by etl/load_bigquery.py). These are what Looker Studio / any
-- BI tool would query directly, instead of hitting the raw table.
--
-- Run these in the BigQuery console, or via `bq query` / the Python client,
-- after load_bigquery.py has populated the dataset.

-- 1. Monthly revenue + returns summary
CREATE OR REPLACE VIEW `{project}.{dataset}.v_monthly_summary` AS
SELECT
  FORMAT_DATE('%Y-%m', Order_Date) AS year_month,
  COUNT(*) AS total_orders,
  SUM(Sales_Amount) AS total_revenue,
  SUM(Profit) AS total_profit,
  SAFE_DIVIDE(SUM(Profit), SUM(Sales_Amount)) * 100 AS overall_margin_percent,
  SUM(Is_Return) AS total_returns,
  SAFE_DIVIDE(SUM(Is_Return), COUNT(*)) * 100 AS return_rate_percent
FROM `{project}.{dataset}.sales_transactions_clean`
GROUP BY year_month
ORDER BY year_month;

-- 2. Performance by product category
CREATE OR REPLACE VIEW `{project}.{dataset}.v_category_performance` AS
SELECT
  Product_Category,
  COUNT(*) AS total_orders,
  SUM(Sales_Amount) AS total_revenue,
  AVG(Margin_Percent) AS avg_margin_percent,
  SAFE_DIVIDE(SUM(Is_Return), COUNT(*)) * 100 AS return_rate_percent
FROM `{project}.{dataset}.sales_transactions_clean`
GROUP BY Product_Category
ORDER BY total_revenue DESC;

-- 3. Channel performance
CREATE OR REPLACE VIEW `{project}.{dataset}.v_channel_performance` AS
SELECT
  Sales_Channel,
  COUNT(*) AS total_orders,
  SUM(Sales_Amount) AS total_revenue,
  AVG(Sales_Amount) AS avg_order_value,
  AVG(Delivery_Days) AS avg_delivery_days
FROM `{project}.{dataset}.sales_transactions_clean`
GROUP BY Sales_Channel
ORDER BY total_revenue DESC;

-- 4. Geographic performance
CREATE OR REPLACE VIEW `{project}.{dataset}.v_geo_performance` AS
SELECT
  Country,
  Region,
  COUNT(*) AS total_orders,
  SUM(Sales_Amount) AS total_revenue,
  SAFE_DIVIDE(SUM(Is_Return), COUNT(*)) * 100 AS return_rate_percent
FROM `{project}.{dataset}.sales_transactions_clean`
GROUP BY Country, Region
ORDER BY total_revenue DESC;

-- 5. Daily revenue series (what feeds the forecasting model)
CREATE OR REPLACE VIEW `{project}.{dataset}.v_daily_revenue` AS
SELECT
  Order_Date AS date,
  SUM(Sales_Amount) AS revenue,
  COUNT(*) AS orders
FROM `{project}.{dataset}.sales_transactions_clean`
GROUP BY date
ORDER BY date;
