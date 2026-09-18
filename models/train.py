"""
Train a PyTorch model to predict whether an order will be returned.

Target: Is_Return (binary)
This is an imbalanced classification problem (~10.4% positive rate), so:
  - We report precision/recall/F1/ROC-AUC, NOT just accuracy (a model predicting
    "never returned" would score ~90% accuracy while being useless).
  - We use a weighted loss (pos_weight) so the model doesn't just learn to
    always predict the majority class.

Usage:
    python models/train.py --data data/processed/sales_clean.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import joblib

NUMERIC_FEATURES = [
    "Customer_Age", "Quantity", "Unit_Price", "Discount_Percentage",
    "Sales_Amount", "Margin_Percent", "Delivery_Days", "Customer_Rating",
]
CATEGORICAL_FEATURES = [
    "Sales_Channel", "Product_Category", "Payment_Method", "Region",
    "Customer_Segment", "Discount_Bucket", "Had_Promotion", "Is_Weekend_Order",
]
TARGET = "Is_Return"


class ReturnRiskNet(nn.Module):
    """Small feedforward net — this is a tabular problem with ~30-40 input
    features after encoding, so a deep/wide architecture isn't warranted.
    Two hidden layers with dropout is enough to model feature interactions
    without overfitting an 18k-row dataset."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET]).copy()

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("Unknown")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return X, y


def train(X_train, y_train, input_dim: int, epochs: int = 60, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Weight the minority (returned) class so the model can't just predict "No" every time
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model = ReturnRiskNet(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    return model, device


def evaluate(model, device, X_test, y_test, threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(X_t)).cpu().numpy().flatten()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "positive_rate_in_test": round(float(y_test.mean()), 4),
    }
    print(classification_report(y_test, preds, target_names=["Not Returned", "Returned"]))
    print("ROC-AUC:", metrics["roc_auc"])
    return metrics


def run(data_path: str, model_dir: str):
    X, y = load_data(data_path)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    model, device = train(X_train, y_train, input_dim=X_train.shape[1])
    metrics = evaluate(model, device, X_test, y_test)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/return_risk_model.pt")
    joblib.dump(preprocessor, f"{model_dir}/preprocessor.joblib")
    with open(f"{model_dir}/input_dim.json", "w") as f:
        json.dump({"input_dim": X_train.shape[1]}, f)
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + preprocessor to {model_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train return-prediction model")
    parser.add_argument("--data", default="data/processed/sales_clean.csv")
    parser.add_argument("--model_dir", default="models/artifacts")
    args = parser.parse_args()
    run(args.data, args.model_dir)
