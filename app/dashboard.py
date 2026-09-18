"""
Interactive dashboard for the sales pipeline project.

Three tabs:
  1. Data Quality — before/after ETL report (what was fixed and why)
  2. Business Overview — revenue, returns, channel/category breakdowns
  3. Revenue Forecast — the LSTM model, with a rolling forecast the user can
     extend into the future

Run:
    streamlit run app/dashboard.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA_CLEAN = ROOT / "data/processed/sales_clean.csv"
QUALITY_REPORT = ROOT / "data/processed/data_quality_report.json"
MODEL_DIR = ROOT / "models/artifacts"

LOOKBACK = 30


class RevenueForecastLSTM(nn.Module):
    def __init__(self, n_series_features: int, n_calendar_features: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_series_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_calendar_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq, x_cal):
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]
        combined = torch.cat([h_last, x_cal], dim=1)
        return self.head(combined)


@st.cache_data
def load_clean_data():
    return pd.read_csv(DATA_CLEAN, parse_dates=["Order_Date"])


@st.cache_data
def load_quality_report():
    with open(QUALITY_REPORT) as f:
        return json.load(f)


@st.cache_resource
def load_forecast_model():
    model = RevenueForecastLSTM(n_series_features=1, n_calendar_features=4)
    model.load_state_dict(torch.load(MODEL_DIR / "revenue_forecast_model.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load(MODEL_DIR / "revenue_scaler.joblib")
    with open(MODEL_DIR / "forecast_metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, metrics


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("Order_Date")["Sales_Amount"].sum().asfreq("D").fillna(0.0)
    daily = daily.reset_index()
    daily.columns = ["date", "revenue"]
    return daily


def forecast_forward(model, scaler, daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Recursively forecast `horizon` days beyond the end of `daily`, feeding
    each prediction back in as input for the next step."""
    history = scaler.transform(daily[["revenue"]]).flatten().tolist()
    last_date = daily["date"].max()

    future_rows = []
    with torch.no_grad():
        for step in range(horizon):
            target_date = last_date + pd.Timedelta(days=step + 1)
            window = np.array(history[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)

            dow = target_date.dayofweek
            doy = target_date.dayofyear
            cal = np.array([[
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
            ]], dtype=np.float32)

            pred_scaled = model(torch.tensor(window), torch.tensor(cal)).item()
            history.append(pred_scaled)

            pred_revenue = scaler.inverse_transform([[pred_scaled]])[0][0]
            future_rows.append({"date": target_date, "forecast_revenue": max(pred_revenue, 0)})

    return pd.DataFrame(future_rows)


st.set_page_config(page_title="Sales Pipeline Portfolio Project", layout="wide")
st.title("Sales Analytics Pipeline")
st.caption("ETL → BigQuery → PyTorch forecasting, built on real messy transaction data")

df = load_clean_data()
quality = load_quality_report()

tab1, tab2, tab3 = st.tabs(["Data Quality (ETL)", "Business Overview", "Revenue Forecast (PyTorch)"])

with tab1:
    st.subheader("What the ETL step fixed")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows before -> after", f"{quality['rows_before']:,} -> {quality['rows_after']:,}")
    col2.metric("Duplicate IDs removed", quality["duplicate_transaction_ids_removed"])
    col3.metric("Implausible ages nulled", quality["implausible_ages_nulled"])

    st.markdown("**Raw data quality issues found and fixed:**")
    st.markdown(
        "- Inconsistent text casing across `Payment_Method`, `Product_Category`, `Order_Status` "
        "(e.g. `'paypal'`, `'PayPal'`, `'PAYPAL'` all normalized to one value)\n"
        "- 45 duplicate `Transaction_ID`s removed\n"
        "- Customer ages outside a plausible range (e.g. 4, 112) nulled rather than silently kept\n"
        "- 9 new analysis-ready fields derived: `Is_Return`, `Margin_Percent`, `Discount_Bucket`, "
        "`Delivery_Speed_Bucket`, `Is_Weekend_Order`, and more"
    )

    st.subheader("Remaining missing values (after cleaning)")
    missing = pd.Series(quality["missing_values_after"]).sort_values(ascending=False)
    st.bar_chart(missing)
    st.caption(
        "Most of this is expected, not a data quality problem: e.g. `Return_Reason` is only "
        "populated when an order was actually returned, and `Promotion_Code` only when a promo was used."
    )

with tab2:
    st.subheader("Business overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total revenue", f"${df['Sales_Amount'].sum():,.0f}")
    col2.metric("Total transactions", f"{len(df):,}")
    col3.metric("Return rate", f"{df['Is_Return'].mean()*100:.1f}%")
    col4.metric("Avg order value", f"${df['Sales_Amount'].mean():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Revenue by product category**")
        st.bar_chart(df.groupby("Product_Category")["Sales_Amount"].sum())
    with c2:
        st.markdown("**Revenue by sales channel**")
        st.bar_chart(df.groupby("Sales_Channel")["Sales_Amount"].sum())

    st.markdown("**Monthly revenue trend (note the Nov/Dec seasonality every year)**")
    monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales_Amount"].sum()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)

    st.markdown("**Revenue by country**")
    st.bar_chart(df.groupby("Country")["Sales_Amount"].sum().sort_values(ascending=False))

with tab3:
    st.subheader("Revenue forecast — LSTM trained on daily revenue")
    model, scaler, metrics = load_forecast_model()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test MAE", f"${metrics['mae']:,.0f}")
    col2.metric("Test RMSE", f"${metrics['rmse']:,.0f}")
    col3.metric("Naive baseline MAE", f"${metrics['naive_baseline_mae']:,.0f}")
    col4.metric("Beats naive baseline", "Yes" if metrics["beats_naive_baseline"] else "No")

    st.caption(
        "Naive baseline = 'tomorrow's revenue will equal the same day last week.' "
        "A forecasting model only earns its place if it beats this."
    )

    horizon = st.slider("Days to forecast into the future", min_value=7, max_value=90, value=30, step=7)

    daily = build_daily_series(df)
    forecast_df = forecast_forward(model, scaler, daily, horizon)

    history_tail = daily.tail(90).rename(columns={"revenue": "value"})
    history_tail["type"] = "actual"
    future = forecast_df.rename(columns={"forecast_revenue": "value"})
    future["type"] = "forecast"

    combined = pd.concat([history_tail[["date", "value", "type"]], future[["date", "value", "type"]]])
    pivot = combined.pivot(index="date", columns="type", values="value")
    st.line_chart(pivot)

    st.markdown(f"**Forecast total for next {horizon} days:** ${forecast_df['forecast_revenue'].sum():,.0f}")
    st.dataframe(forecast_df.style.format({"forecast_revenue": "${:,.0f}"}), use_container_width=True)
