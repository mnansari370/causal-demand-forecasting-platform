from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.data.preprocess import (
    add_lag_features,
    add_rolling_features,
    save_parquet,
)
from src.utils.logger import get_logger


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)

    logger = get_logger(
        "build_forecasting_features",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]

    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    logger.info("Loading processed train/val/test parquet files")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Val shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info("Combined shape before feature engineering: %s", full_df.shape)

    target_col = config["data"]["target_column"]
    group_cols = [config["data"]["store_column"], config["data"]["item_column"]]
    lag_days = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]

    full_df = add_lag_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        lag_days=lag_days,
    )

    full_df = add_rolling_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        windows=rolling_windows,
    )

    date_col = config["data"]["date_column"]
    splits = config["splits"]

    train_feat = full_df[full_df[date_col] <= pd.to_datetime(splits["train_end"])].copy()
    val_feat = full_df[
        (full_df[date_col] >= pd.to_datetime(splits["val_start"])) &
        (full_df[date_col] <= pd.to_datetime(splits["val_end"]))
    ].copy()
    test_feat = full_df[
        (full_df[date_col] >= pd.to_datetime(splits["test_start"])) &
        (full_df[date_col] <= pd.to_datetime(splits["test_end"]))
    ].copy()

    logger.info("Train features shape: %s", train_feat.shape)
    logger.info("Val features shape: %s", val_feat.shape)
    logger.info("Test features shape: %s", test_feat.shape)

    save_parquet(train_feat, processed_dir / "train_features.parquet")
    save_parquet(val_feat, processed_dir / "val_features.parquet")
    save_parquet(test_feat, processed_dir / "test_features.parquet")

    logger.info("Forecasting feature build complete")


if __name__ == "__main__":
    main()