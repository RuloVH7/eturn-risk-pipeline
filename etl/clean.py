"""
ETL: Clean and transform raw sales transaction data.

Raw data quality issues this script addresses (discovered during profiling):
  - Inconsistent text casing in categorical fields (Payment_Method, Product_Category,
    Order_Status) — e.g. 'paypal' / 'PayPal' / 'Credit Card' / 'credit card'
  - 45 duplicate Transaction_IDs
  - Customer_Age outliers (min=4, max=112) — implausible for a retail customer
  - Missing values with different meanings:
      * Return_Reason missing (89.6%) -> expected, only set when Return_Flag = 'Yes'
      * Promotion_Code missing (54.0%) -> expected, no promo applied
      * Customer_Rating missing (19.0%) -> no rating left, NOT the same as a 0 rating
  - Derives features needed for the downstream return-prediction model

Usage:
    python etl/clean.py --input data/raw/Sales_transactions_2022_2025.csv \
                         --output data/processed/sales_clean.csv \
                         --report data/processed/data_quality_report.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Plausible bounds for a retail customer; anything outside is treated as bad data,
# not dropped (we don't want to silently lose transactions), just flagged + nulled.
MIN_PLAUSIBLE_AGE = 12
MAX_PLAUSIBLE_AGE = 95


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Transaction_ID": str, "Order_ID": str, "Customer_ID": str})
    return df


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix casing inconsistencies in categorical/text fields."""
    df = df.copy()

    # Simple title/upper normalization for category-like fields
    for col in ["Product_Category", "Order_Status"]:
        df[col] = df[col].astype(str).str.strip().str.title()
        df.loc[df[col] == "Nan", col] = np.nan

    # Payment_Method needs explicit mapping — title-casing alone won't fix
    # 'paypal' -> 'Paypal' vs the desired 'PayPal', or 'DebitCard' -> 'Debit Card'
    payment_map = {
        "credit card": "Credit Card",
        "debit card": "Debit Card",
        "debitcard": "Debit Card",
        "paypal": "PayPal",
        "apple pay": "Apple Pay",
        "cash": "Cash",
        "bank transfer": "Bank Transfer",
    }
    df["Payment_Method"] = (
        df["Payment_Method"].astype(str).str.strip().str.lower().map(payment_map)
    )

    for col in ["Sales_Channel", "Country", "Region", "City", "Return_Flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["nan", "Nan", "NaN", ""]), col] = np.nan

    return df


def dedupe_transactions(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    before = len(df)
    dupe_mask = df.duplicated(subset=["Transaction_ID"], keep="first")
    report["duplicate_transaction_ids_removed"] = int(dupe_mask.sum())
    df = df.loc[~dupe_mask].copy()
    report["rows_after_dedupe"] = len(df)
    report["rows_dropped_dedupe"] = before - len(df)
    return df


def fix_age_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    df = df.copy()
    bad_age_mask = (df["Customer_Age"] < MIN_PLAUSIBLE_AGE) | (df["Customer_Age"] > MAX_PLAUSIBLE_AGE)
    report["implausible_ages_nulled"] = int(bad_age_mask.sum())
    df.loc[bad_age_mask, "Customer_Age"] = np.nan
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S", errors="coerce").dt.time
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fields useful for BI + the return-prediction model."""
    df = df.copy()

    df["Order_DayOfWeek"] = df["Order_Date"].dt.day_name()
    df["Is_Weekend_Order"] = df["Order_Date"].dt.dayofweek.isin([5, 6])
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Quarter"] = df["Order_Date"].dt.quarter

    df["Had_Promotion"] = df["Promotion_Code"].notna()
    df["Discount_Bucket"] = pd.cut(
        df["Discount_Percentage"].fillna(0),
        bins=[-0.01, 0, 10, 20, 100],
        labels=["None", "Low (0-10%)", "Medium (10-20%)", "High (20%+)"],
    )

    df["Margin_Percent"] = np.where(
        df["Sales_Amount"] > 0, (df["Profit"] / df["Sales_Amount"]) * 100, np.nan
    )

    df["Is_Return"] = (df["Return_Flag"] == "Yes").astype(int)

    df["Delivery_Speed_Bucket"] = pd.cut(
        df["Delivery_Days"].fillna(-1),
        bins=[-2, -1, 2, 5, 100],
        labels=["Unknown/Not Shipped", "Fast (<=2d)", "Standard (3-5d)", "Slow (6d+)"],
    )

    return df


def build_report(df_before: pd.DataFrame, df_after: pd.DataFrame, report: dict) -> dict:
    report["rows_before"] = len(df_before)
    report["rows_after"] = len(df_after)
    report["columns_before"] = df_before.shape[1]
    report["columns_after"] = df_after.shape[1]
    report["return_rate_percent"] = round(float(df_after["Is_Return"].mean() * 100), 2)
    report["missing_values_after"] = {
        col: int(df_after[col].isna().sum())
        for col in df_after.columns
        if df_after[col].isna().sum() > 0
    }
    return report


def run(input_path: str, output_path: str, report_path: str):
    report = {}

    df = load_raw(input_path)
    df_before = df.copy()

    df = standardize_text_columns(df)
    df = dedupe_transactions(df, report)
    df = fix_age_outliers(df, report)
    df = parse_dates(df)
    df = derive_features(df)

    report = build_report(df_before, df, report)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Cleaned {report['rows_before']} -> {report['rows_after']} rows")
    print(f"Removed {report['duplicate_transaction_ids_removed']} duplicate transaction IDs")
    print(f"Nulled {report['implausible_ages_nulled']} implausible ages")
    print(f"Return rate: {report['return_rate_percent']}%")
    print(f"Saved cleaned data to: {output_path}")
    print(f"Saved data quality report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw sales transaction data")
    parser.add_argument("--input", default="data/raw/Sales_transactions_2022_2025.csv")
    parser.add_argument("--output", default="data/processed/sales_clean.csv")
    parser.add_argument("--report", default="data/processed/data_quality_report.json")
    args = parser.parse_args()
    run(args.input, args.output, args.report)
