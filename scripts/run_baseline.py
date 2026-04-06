from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.evaluation.metrics import rmse, mae, mape
from src.utils.logger import get_logger


def evaluate_split(
    split_df: pd.DataFrame,
    split_name: str,
    target_col: str,
    lag_col: str,
    global_mean: float,
    logger,
) -> dict:
    if lag_col not in split_df.columns:
        logger.error("Lag feature %s not found in %s split", lag_col, split_name)
        sys.exit(1)

    preds = split_df[lag_col].copy()
    fallback_mask = preds.isna()
    fallback_count = int(fallback_mask.sum())

    if fallback_count > 0:
        logger.info(
            "Fallback to global mean for %d rows in %s (missing %s)",
            fallback_count,
            split_name,
            lag_col,
        )
        preds.loc[fallback_mask] = global_mean

    y_true = split_df[target_col].values
    y_pred = preds.values

    metrics = {
        "model": "Seasonal Naive (S=7)",
        "evaluation_split": split_name,
        "history_source": lag_col,
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
        "probabilistic": False,
        "coverage_90": None,
        "fallback_global_mean_count": fallback_count,
    }

    logger.info(
        "Seasonal Naive [%s] -> RMSE=%.4f | MAE=%.4f | MAPE=%.2f%% | fallback=%d",
        split_name,
        metrics["rmse"],
        metrics["mae"],
        metrics["mape"],
        fallback_count,
    )

    return metrics


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
    test_path = processed_dir / "test_features.parquet"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        logger.error("Feature parquets not found. Run scripts/build_forecasting_features.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Validation shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    target_col = config["data"]["target_column"]
    lag_col = f"{target_col}_lag_7"
    global_mean = train_df[target_col].mean()

    logger.info("Using %s as Seasonal Naive (S=7) prediction", lag_col)

    results = []
    results.append(evaluate_split(val_df, "validation", target_col, lag_col, global_mean, logger))
    results.append(evaluate_split(test_df, "test", target_col, lag_col, global_mean, logger))

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "week1_baseline_results.csv", index=False)

    with open(results_dir / "week1_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("WEEK 1 BASELINE RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    print("\nUse validation results for model comparison during Week 2.")
    print("Use test results for final reporting later.")


if __name__ == "__main__":
    main()