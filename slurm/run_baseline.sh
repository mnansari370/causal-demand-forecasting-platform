from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.utils.logger import get_logger


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100)


def seasonal_naive_predict(history_df, target_df, config, seasonality=7):
    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]

    lookup = history_df.set_index([store_col, item_col, date_col])[target_col].to_dict()
    global_mean = history_df[target_col].mean()

    predictions = []
    misses = 0

    for _, row in target_df.iterrows():
        hist_date = row[date_col] - pd.Timedelta(days=seasonality)
        key = (row[store_col], row[item_col], hist_date)
        pred = lookup.get(key, None)

        if pred is None:
            pred = global_mean
            misses += 1

        predictions.append(pred)

    return np.array(predictions), misses


def main():
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    logger = get_logger(
        "run_baseline",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = processed_dir / "train_features.parquet"
    target_path = processed_dir / "val_features.parquet"

    if not history_path.exists() or not target_path.exists():
        logger.error("Required feature parquets not found. Run scripts/build_forecasting_features.py first.")
        sys.exit(1)

    history_df = pd.read_parquet(history_path)
    target_df = pd.read_parquet(target_path)

    logger.info("History shape (train): %s", history_df.shape)
    logger.info("Target shape (validation): %s", target_df.shape)

    target_col = config["data"]["target_column"]

    logger.info("Running Seasonal Naive baseline on validation with seasonality=7")
    preds, misses = seasonal_naive_predict(history_df, target_df, config, seasonality=7)

    y_true = target_df[target_col].values

    metrics = {
        "model": "Seasonal Naive (S=7)",
        "evaluation_split": "validation",
        "history_split": "train",
        "rmse": round(rmse(y_true, preds), 4),
        "mae": round(mae(y_true, preds), 4),
        "mape": round(mape(y_true, preds), 4),
        "probabilistic": False,
        "coverage_90": None,
        "fallback_global_mean_count": int(misses),
    }

    logger.info("Seasonal Naive validation results: %s", metrics)

    results_df = pd.DataFrame([metrics])
    results_df.to_csv(results_dir / "week1_baseline_results.csv", index=False)

    with open(results_dir / "week1_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump([metrics], f, indent=2)

    print("\n" + "=" * 60)
    print("WEEK 1 BASELINE RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()