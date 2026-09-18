"""
Simulate new transactions "arriving" each day, so the n8n daily pipeline has
something real to process instead of reprocessing a static, unchanging file.

How it works:
  - Tracks the last simulated date in data/raw/last_generated_date.txt
  - Each run advances by `--days` days (default 1), generating a realistic
    batch of transactions for each new day
  - New rows are sampled from the REAL historical distributions (categories,
    prices, regions, channels stay statistically consistent with the actual
    dataset) rather than invented from scratch
  - Appends to the raw CSV; the existing ETL step reprocesses the whole file
    on each run (cheap at this scale — ~18k rows takes well under a second)

This is a portfolio/demo device, not a claim that real transactions are
arriving — see README for how this is framed honestly.

Usage:
    python etl/generate_daily_batch.py --days 1
    python etl/generate_daily_batch.py --days 7    # catch up a week at once
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/Sales_transactions_2022_2025.csv"
STATE_PATH = "data/raw/last_generated_date.txt"
AVG_DAILY_TRANSACTIONS = 12.35  # matches the historical average (18,045 rows / 1,461 days)


def get_last_date(df: pd.DataFrame) -> pd.Timestamp:
    state_file = Path(STATE_PATH)
    if state_file.exists():
        return pd.Timestamp(state_file.read_text().strip())
    return pd.to_datetime(df["Order_Date"]).max()


def next_id(df: pd.DataFrame, col: str, prefix: str) -> int:
    return int(df[col].str.extract(r"(\d+)")[0].astype(int).max())


def generate_batch(df: pd.DataFrame, target_date: pd.Timestamp, start_txn_id: int, start_order_id: int, rng: np.random.Generator) -> pd.DataFrame:
    n = max(1, rng.poisson(AVG_DAILY_TRANSACTIONS))

    # Sample whole rows to keep realistic combos (e.g. a Store_ID always
    # pairs with the right Store_Name/City/Country; a Product_ID always
    # pairs with the right Category/Subcategory/Unit_Price)
    sampled = df.sample(n=n, replace=True, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)

    new_rows = sampled.copy()
    new_rows["Transaction_ID"] = [f"T{start_txn_id + i:07d}" for i in range(n)]
    new_rows["Order_ID"] = [f"O{start_order_id + i:07d}" for i in range(n)]
    new_rows["Order_Date"] = target_date.strftime("%Y-%m-%d")
    new_rows["Order_Time"] = [
        f"{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:{rng.integers(0,60):02d}" for _ in range(n)
    ]
    new_rows["Order_Year"] = target_date.year

    # Re-roll the fields that should vary day to day rather than being
    # copied verbatim from the sampled historical row — sampled FROM the
    # real historical distribution, not an invented range, so the new
    # transactions stay statistically consistent with the real dataset
    new_rows["Quantity"] = df["Quantity"].sample(n=n, replace=True, random_state=rng.integers(0, 1_000_000)).values
    new_rows["Discount_Percentage"] = df["Discount_Percentage"].sample(n=n, replace=True, random_state=rng.integers(0, 1_000_000)).values
    new_rows["Sales_Amount"] = (
        new_rows["Unit_Price"] * new_rows["Quantity"] * (1 - new_rows["Discount_Percentage"] / 100)
    ).round(2)
    # Keep the historical cost-to-revenue ratio distribution rather than a fixed margin
    cost_ratio = (df["Cost_Amount"] / df["Sales_Amount"].replace(0, np.nan)).dropna().sample(n=n, replace=True, random_state=rng.integers(0, 1_000_000)).values
    new_rows["Cost_Amount"] = (new_rows["Sales_Amount"] * cost_ratio).round(2)
    new_rows["Profit"] = (new_rows["Sales_Amount"] - new_rows["Cost_Amount"]).round(2)

    # Return_Flag / Customer_Rating kept as independent random draws, consistent
    # with the earlier finding that these carry no real signal in this dataset
    new_rows["Return_Flag"] = rng.choice(["No", "Yes"], size=n, p=[0.896, 0.104])
    reason_choices = rng.choice(["Damaged", "Wrong Item", "Changed Mind", "Defective"], size=n)
    new_rows["Return_Reason"] = pd.Series(reason_choices, dtype="object").where(new_rows["Return_Flag"] == "Yes", other=np.nan)

    return new_rows


def run(days: int, seed: int | None):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(RAW_PATH, dtype={"Transaction_ID": str, "Order_ID": str, "Customer_ID": str})

    last_date = get_last_date(df)
    next_txn_id = next_id(df, "Transaction_ID", "T") + 1
    next_order_id = next_id(df, "Order_ID", "O") + 1

    all_new = []
    current_date = last_date
    for _ in range(days):
        current_date = current_date + pd.Timedelta(days=1)
        batch = generate_batch(df, current_date, next_txn_id, next_order_id, rng)
        all_new.append(batch)
        next_txn_id += len(batch)
        next_order_id += len(batch)

    new_df = pd.concat(all_new, ignore_index=True)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(RAW_PATH, index=False)

    Path(STATE_PATH).write_text(current_date.strftime("%Y-%m-%d"))

    print(f"Generated {len(new_df)} new transactions across {days} day(s)")
    print(f"Date range added: {last_date.strftime('%Y-%m-%d')} -> {current_date.strftime('%Y-%m-%d')}")
    print(f"Raw dataset now has {len(combined)} total rows (was {len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate new daily transactions arriving")
    parser.add_argument("--days", type=int, default=1, help="How many new days of data to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    run(args.days, args.seed)
