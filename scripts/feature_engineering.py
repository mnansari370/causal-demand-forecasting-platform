"""
Build lag and rolling forecasting features.

Features are computed on the concatenated train + val + test dataframe and
then split back into train/val/test. This is the correct approach for
time-series because validation rows need lag history from train, and test
rows need lag history from validation.

If we computed features separately inside each split, the early rows in each
split would lose valid historical context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.data.preprocess import add_lag_features, add_rolling_features, save_parquet
from src.utils.logger import get_logger


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "feature_engineering",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]

    for split in ["train", "val", "test"]:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            logger.error("%s not found. Run scripts/data_preparation.py first.", path)
            sys.exit(1)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Shapes | train=%s | val=%s | test=%s", train_df.shape, val_df.shape, test_df.shape)

    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    group_cols = [store_col, item_col]

    full_df = (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values(group_cols + [date_col])
        .reset_index(drop=True)
    )

    logger.info("Adding lag features")
    full_df = add_lag_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        lag_days=config["features"]["lag_days"],
    )

    logger.info("Adding rolling features")
    full_df = add_rolling_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        windows=config["features"]["rolling_windows"],
    )

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

    logger.info(
        "Feature split shapes | train=%s | val=%s | test=%s",
        train_feat.shape,
        val_feat.shape,
        test_feat.shape,
    )

    save_parquet(train_feat, processed_dir / "train_features.parquet")
    save_parquet(val_feat, processed_dir / "val_features.parquet")
    save_parquet(test_feat, processed_dir / "test_features.parquet")

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    main()