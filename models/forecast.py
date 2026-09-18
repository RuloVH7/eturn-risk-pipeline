"""
Forecast daily revenue using an LSTM, trained on aggregated daily sales.

Why this target (validated before building, see data/processed/seasonality_check.json):
  - Monthly revenue shows a clear, consistent holiday seasonality (Nov/Dec revenue
    is ~2x January revenue, every single year from 2022-2025) plus mild YoY growth.
  - This is real signal, unlike Is_Return / Customer_Rating in this dataset, which
    were confirmed to be statistically unrelated to any other column (see
    models/train.py — kept in the repo as a documented negative result).

Approach:
  - Aggregate transactions to a daily revenue series (1,461 days, 2022-01-01 to 2025-12-31)
  - Feature the series with a lookback window + calendar features (day-of-week,
    day-of-year, month) so the model can learn both short-term momentum and
    yearly seasonality
  - Train an LSTM, evaluate on the last 90 days (held out chronologically —
    a random split would leak future information into training, which is
    invalid for time series)

Usage:
    python models/forecast.py --data data/processed/sales_clean.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOOKBACK = 30       # days of history the model sees to make one prediction
TEST_DAYS = 90       # holdout period at the end of the series
EPOCHS = 200
LR = 1e-3


class RevenueForecastLSTM(nn.Module):
    """LSTM over the lookback window, with calendar features concatenated
    at the final step. LSTM is the right tool here (vs. a plain feedforward
    net) because revenue on a given day depends on the *sequence* of recent
    days, not just their independent values — momentum and trend matter."""

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
        h_last = h_n[-1]  # (batch, hidden_size)
        combined = torch.cat([h_last, x_cal], dim=1)
        return self.head(combined)


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("Order_Date")["Sales_Amount"].sum().asfreq("D").fillna(0.0)
    daily = daily.reset_index()
    daily.columns = ["date", "revenue"]
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["is_month_end"] = daily["date"].dt.is_month_end.astype(int)
    return daily


def make_sequences(daily: pd.DataFrame, scaler: StandardScaler):
    """Build (lookback-window, calendar-features-at-target-day) -> next-day-revenue samples."""
    revenue_scaled = scaler.transform(daily[["revenue"]]).flatten()

    # Sin/cos encode cyclical calendar features so e.g. Dec 31 -> Jan 1 is continuous
    dow_sin = np.sin(2 * np.pi * daily["dow"] / 7)
    dow_cos = np.cos(2 * np.pi * daily["dow"] / 7)
    doy_sin = np.sin(2 * np.pi * daily["day_of_year"] / 365)
    doy_cos = np.cos(2 * np.pi * daily["day_of_year"] / 365)

    X_seq, X_cal, y = [], [], []
    for i in range(LOOKBACK, len(daily)):
        X_seq.append(revenue_scaled[i - LOOKBACK:i].reshape(-1, 1))
        X_cal.append([dow_sin[i], dow_cos[i], doy_sin[i], doy_cos[i]])
        y.append(revenue_scaled[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_cal, dtype=np.float32),
        np.array(y, dtype=np.float32),
        daily["date"].iloc[LOOKBACK:].reset_index(drop=True),
    )


def train(model, X_seq_tr, X_cal_tr, y_tr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_seq_t = torch.tensor(X_seq_tr).to(device)
    X_cal_t = torch.tensor(X_cal_tr).to(device)
    y_t = torch.tensor(y_tr).unsqueeze(1).to(device)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_seq_t, X_cal_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - MSE loss: {loss.item():.4f}")
    return model


def evaluate(model, X_seq_te, X_cal_te, y_te, dates_te, scaler, device):
    model.eval()
    with torch.no_grad():
        X_seq_t = torch.tensor(X_seq_te).to(device)
        X_cal_t = torch.tensor(X_cal_te).to(device)
        preds_scaled = model(X_seq_t, X_cal_t).cpu().numpy().flatten()

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape = float(np.mean(np.abs((actual - preds) / np.clip(actual, 1, None))) * 100)

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape_percent": round(mape, 2),
        "test_days": len(actual),
        "actual_mean_daily_revenue": round(float(actual.mean()), 2),
    }
    print(f"MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | MAPE: {metrics['mape_percent']}%")

    # Naive baseline: "tomorrow = same day last week" — a forecasting model
    # needs to beat this to be worth anything
    naive_preds = actual.copy()
    naive_preds[7:] = actual[:-7]
    naive_mae = mean_absolute_error(actual[7:], naive_preds[7:])
    metrics["naive_baseline_mae"] = round(float(naive_mae), 2)
    metrics["beats_naive_baseline"] = bool(mae < naive_mae)
    print(f"Naive baseline (same day last week) MAE: {metrics['naive_baseline_mae']}")
    print(f"Model beats naive baseline: {metrics['beats_naive_baseline']}")

    predictions_df = pd.DataFrame({"date": dates_te, "actual": actual, "predicted": preds})
    return metrics, predictions_df


def run(data_path: str, model_dir: str):
    df = pd.read_csv(data_path, parse_dates=["Order_Date"])
    daily = build_daily_series(df)

    split_idx = len(daily) - TEST_DAYS
    train_daily = daily.iloc[: split_idx + LOOKBACK]  # include lookback context for test window
    test_daily = daily.iloc[split_idx:]

    scaler = StandardScaler()
    scaler.fit(daily.iloc[:split_idx][["revenue"]])  # fit only on train period, no leakage

    X_seq_tr, X_cal_tr, y_tr, _ = make_sequences(train_daily.reset_index(drop=True), scaler)
    X_seq_te, X_cal_te, y_te, dates_te = make_sequences(daily.iloc[split_idx - LOOKBACK:].reset_index(drop=True), scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RevenueForecastLSTM(n_series_features=1, n_calendar_features=4).to(device)
    model = train(model, X_seq_tr, X_cal_tr, y_tr, device)
    metrics, predictions_df = evaluate(model, X_seq_te, X_cal_te, y_te, dates_te, scaler, device)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/revenue_forecast_model.pt")
    import joblib
    joblib.dump(scaler, f"{model_dir}/revenue_scaler.joblib")
    with open(f"{model_dir}/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    predictions_df.to_csv(f"{model_dir}/forecast_predictions.csv", index=False)

    print(f"\nSaved model + predictions to {model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train daily revenue forecasting model")
    parser.add_argument("--data", default="data/processed/sales_clean.csv")
    parser.add_argument("--model_dir", default="models/artifacts")
    args = parser.parse_args()
    run(args.data, args.model_dir)
