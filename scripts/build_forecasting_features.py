# scripts/build_forecasting_features.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.data.preprocess import add_lag_features, add_rolling_features, save_parquet
from src.utils.logger import get_logger


def log_missing_feature_audit(logger, df: pd.DataFrame, feature_cols: list[str], split_name: str) -> None:
    logger.info("=" * 60)
    logger.info("Missing feature audit for %s", split_name)
    logger.info("=" * 60)

    for col in feature_cols:
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df) if len(df) > 0 else 0.0
        logger.info("%s -> %d missing (%.2f%%)", col, n_missing, pct_missing)


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

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        logger.error("Processed parquet files not found. Run scripts/run_data_check.py first.")
        sys.exit(1)

    logger.info("Loading processed splits")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Val shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]

    lag_days = config["features"]["lag_days"]
    rolling_windows = config["features"]["rolling_windows"]
    group_cols = [store_col, item_col]

    logger.info("Concatenating all splits in chronological order for feature generation")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    logger.info("Adding lag features")
    full_df = add_lag_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        lag_days=lag_days,
    )

    logger.info("Adding rolling features")
    full_df = add_rolling_features(
        full_df,
        target_col=target_col,
        group_cols=group_cols,
        windows=rolling_windows,
    )

    lag_cols = [f"{target_col}_lag_{d}" for d in lag_days]
    rolling_cols = []
    for w in rolling_windows:
        rolling_cols.extend([
            f"{target_col}_rolling_mean_{w}d",
            f"{target_col}_rolling_std_{w}d",
        ])

    feature_cols = lag_cols + rolling_cols

    logger.info("=" * 60)
    logger.info("Global NaN audit before re-splitting")
    logger.info("=" * 60)

    all_nan_lag_mask = full_df[lag_cols].isna().all(axis=1)
    n_all_nan_lag = all_nan_lag_mask.sum()
    logger.info(
        "Rows with all lag features NaN: %d / %d (%.2f%%)",
        n_all_nan_lag,
        len(full_df),
        100 * n_all_nan_lag / len(full_df),
    )

    for col in feature_cols:
        n_missing = full_df[col].isna().sum()
        pct_missing = 100 * n_missing / len(full_df)
        logger.info("%s -> %d missing (%.2f%%)", col, n_missing, pct_missing)

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

    logger.info("Final feature split shapes:")
    logger.info("Train features: %s", train_feat.shape)
    logger.info("Val features: %s", val_feat.shape)
    logger.info("Test features: %s", test_feat.shape)

    log_missing_feature_audit(logger, train_feat, feature_cols, "train")
    log_missing_feature_audit(logger, val_feat, feature_cols, "validation")
    log_missing_feature_audit(logger, test_feat, feature_cols, "test")

    save_parquet(train_feat, processed_dir / "train_features.parquet")
    save_parquet(val_feat, processed_dir / "val_features.parquet")
    save_parquet(test_feat, processed_dir / "test_features.parquet")

    logger.info("Forecasting feature generation complete")


if __name__ == "__main__":
    main()