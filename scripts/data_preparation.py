"""
Prepare the main train, validation, and test datasets.

This script:
1. loads the Favorita data (or the dev subset if enabled),
2. cleans the training table,
3. merges side tables,
4. adds calendar features,
5. performs a strict temporal split,
6. saves the processed parquet files.

All splits are chronological. We never use random splitting for time-series.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config, load_favorita_data, summarize_dataframe
from src.data.preprocess import (
    add_calendar_features,
    clean_train,
    merge_holidays,
    merge_items,
    merge_oil,
    merge_stores,
    save_csv,
    save_parquet,
    temporal_split,
)
from src.utils.logger import get_logger


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "data_preparation",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    logger.info("Loading Favorita data")
    data = load_favorita_data(config)

    for name, df in data.items():
        summarize_dataframe(name, df)

    train_df = data.get("train")
    if train_df is None:
        logger.error("Train data not found")
        sys.exit(1)

    logger.info("Cleaning training data")
    train_df = clean_train(train_df, config)

    logger.info("Merging side tables")
    train_df = merge_stores(train_df, data.get("stores"))
    train_df = merge_items(train_df, data.get("items"))
    train_df = merge_oil(train_df, data.get("oil"), date_col=config["data"]["date_column"])
    train_df = merge_holidays(train_df, data.get("holidays_events"), date_col=config["data"]["date_column"])

    logger.info("Adding calendar features")
    train_df = add_calendar_features(train_df, date_col=config["data"]["date_column"])

    logger.info(
        "Processed shape: %s | memory: %.2f MB",
        train_df.shape,
        train_df.memory_usage(deep=True).sum() / 1e6,
    )

    logger.info("Performing temporal split")
    train_split, val_split, test_split = temporal_split(
        train_df,
        date_col=config["data"]["date_column"],
        splits=config["splits"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    save_parquet(train_split, processed_dir / "train.parquet")
    save_parquet(val_split, processed_dir / "val.parquet")
    save_parquet(test_split, processed_dir / "test.parquet")

    check_dir = PROJECT_ROOT / config["outputs"]["data_check_dir"]
    check_dir.mkdir(parents=True, exist_ok=True)

    save_csv(train_split.head(1000), check_dir / "train_sample.csv")

    report = {
        "train_shape": list(train_split.shape),
        "val_shape": list(val_split.shape),
        "test_shape": list(test_split.shape),
        "columns": list(train_split.columns),
        "missing_values": train_split.isnull().sum()[train_split.isnull().sum() > 0].to_dict(),
        "date_range": {
            "min": str(train_split[config["data"]["date_column"]].min()),
            "max": str(test_split[config["data"]["date_column"]].max()),
        },
    }

    report_path = check_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote data quality report to: %s", report_path)

    logger.info("Data preparation complete")


if __name__ == "__main__":
    main()