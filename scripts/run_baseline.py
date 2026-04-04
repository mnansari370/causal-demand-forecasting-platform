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

    train_path = processed_dir / "train_features.parquet"
    val_path = processed_dir / "val_features.parquet"

    if not train_path.exists() or not val_path.exists():
        logger.error("Required feature parquets not found. Run scripts/build_forecasting_features.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Validation shape: %s", val_df.shape)

    target_col = config["data"]["target_column"]
    lag_col = f"{target_col}_lag_7"

    if lag_col not in val_df.columns:
        logger.error("Required lag feature not found in validation data: %s", lag_col)
        sys.exit(1)

    global_mean = train_df[target_col].mean()

    logger.info("Using %s as Seasonal Naive (S=7) prediction", lag_col)

    preds = val_df[lag_col].copy()
    fallback_mask = preds.isna()
    fallback_count = int(fallback_mask.sum())

    if fallback_count > 0:
        logger.info(
            "Falling back to train global mean for %d validation rows with missing %s",
            fallback_count,
            lag_col,
        )
        preds.loc[fallback_mask] = global_mean

    y_true = val_df[target_col].values
    y_pred = preds.values

    metrics = {
        "model": "Seasonal Naive (S=7)",
        "evaluation_split": "validation",
        "history_source": lag_col,
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
        "probabilistic": False,
        "coverage_90": None,
        "fallback_global_mean_count": fallback_count,
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
