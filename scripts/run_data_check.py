from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config, load_favorita_data, summarize_dataframe
from src.data.preprocess import (
    clean_train,
    merge_stores,
    merge_items,
    merge_oil,
    merge_holidays,
    add_calendar_features,
    temporal_split,
    save_parquet,
    save_csv,
)
from src.utils.logger import get_logger


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)

    logger = get_logger(
        "run_data_check",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    logger.info("=" * 60)
    logger.info("STEP 1: Loading Favorita data")
    logger.info("=" * 60)

    data = load_favorita_data(config)

    for name, df in data.items():
        summarize_dataframe(name, df)

    train_df = data.get("train")
    if train_df is None:
        logger.error("train.csv not found in %s", config["data"]["raw_data_dir"])
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning train data")
    logger.info("=" * 60)
    train_df = clean_train(train_df, config)

    logger.info("=" * 60)
    logger.info("STEP 3: Merging side tables")
    logger.info("=" * 60)
    train_df = merge_stores(train_df, data.get("stores"))
    train_df = merge_items(train_df, data.get("items"))
    train_df = merge_oil(train_df, data.get("oil"), date_col=config["data"]["date_column"])
    train_df = merge_holidays(train_df, data.get("holidays_events"), date_col=config["data"]["date_column"])

    logger.info("=" * 60)
    logger.info("STEP 4: Adding calendar features")
    logger.info("=" * 60)
    train_df = add_calendar_features(train_df, date_col=config["data"]["date_column"])

    logger.info("Processed shape after feature enrichment: %s", train_df.shape)
    logger.info("Memory usage: %.2f MB", train_df.memory_usage(deep=True).sum() / 1e6)

    logger.info("=" * 60)
    logger.info("STEP 5: Temporal split")
    logger.info("=" * 60)
    train_split, val_split, test_split = temporal_split(
        train_df,
        date_col=config["data"]["date_column"],
        splits=config["splits"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    check_dir = PROJECT_ROOT / config["outputs"]["data_check_dir"]
    check_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 6: Saving outputs")
    logger.info("=" * 60)
    save_parquet(train_split, processed_dir / "train.parquet")
    save_parquet(val_split, processed_dir / "val.parquet")
    save_parquet(test_split, processed_dir / "test.parquet")

    save_csv(train_split.head(1000), check_dir / "train_sample_1k.csv")

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
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote data quality report to: %s", report_path)

    logger.info("=" * 60)
    logger.info("Data check complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()