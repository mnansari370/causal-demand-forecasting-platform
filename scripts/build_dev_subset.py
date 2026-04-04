from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.utils.logger import get_logger


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)

    logger = get_logger(
        "build_dev_subset",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    raw_dir = PROJECT_ROOT / config["data"]["raw_data_dir"]
    interim_dir = PROJECT_ROOT / config["data"]["interim_data_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    train_file = raw_dir / config["data"]["files"]["train"]
    if not train_file.exists():
        logger.error("Raw train file not found: %s", train_file)
        sys.exit(1)

    subset_cfg = config["dev_subset"]
    start_date = pd.to_datetime(subset_cfg["subset_start_date"])
    end_date = pd.to_datetime(subset_cfg["subset_end_date"])
    rows_per_day = subset_cfg["rows_per_day"]
    chunk_size = subset_cfg["chunk_size"]
    output_file = interim_dir / subset_cfg["output_file"]

    logger.info("=" * 60)
    logger.info("Building time-balanced development subset")
    logger.info("Input file: %s", train_file)
    logger.info("Date window: %s to %s", start_date.date(), end_date.date())
    logger.info("Rows per day: %s", rows_per_day)
    logger.info("Chunk size: %s", chunk_size)
    logger.info("=" * 60)

    selected_chunks = []
    total_rows_seen = 0
    daily_counts: dict[pd.Timestamp, int] = {}

    for i, chunk in enumerate(
        pd.read_csv(
            train_file,
            chunksize=chunk_size,
            parse_dates=[config["data"]["date_column"]],
            dtype={"onpromotion": "object"},
        ),
        start=1,
    ):
        total_rows_seen += len(chunk)

        # Filter date window first
        chunk = chunk[
            (chunk[config["data"]["date_column"]] >= start_date) &
            (chunk[config["data"]["date_column"]] <= end_date)
        ]

        if chunk.empty:
            continue

        kept_parts = []

        for day, day_df in chunk.groupby(config["data"]["date_column"]):
            current_count = daily_counts.get(day, 0)
            remaining = rows_per_day - current_count

            if remaining <= 0:
                continue

            take_df = day_df.iloc[:remaining]
            if not take_df.empty:
                kept_parts.append(take_df)
                daily_counts[day] = current_count + len(take_df)

        if kept_parts:
            kept_chunk = pd.concat(kept_parts, ignore_index=True)
            selected_chunks.append(kept_chunk)

            logger.info(
                "Chunk %d: kept %d rows | total seen=%d | unique days covered=%d",
                i,
                len(kept_chunk),
                total_rows_seen,
                len(daily_counts),
            )

        # Stop early if every day in the window is already filled
        total_days_needed = (end_date - start_date).days + 1
        if len(daily_counts) == total_days_needed and all(v >= rows_per_day for v in daily_counts.values()):
            logger.info("All days in requested window have reached rows_per_day. Stopping.")
            break

    if not selected_chunks:
        logger.error("No rows found in the requested date range.")
        sys.exit(1)

    subset_df = pd.concat(selected_chunks, ignore_index=True)
    subset_df = subset_df.sort_values(config["data"]["date_column"]).reset_index(drop=True)

    logger.info("Final subset shape: %s", subset_df.shape)
    logger.info(
        "Subset date range: %s to %s",
        subset_df[config["data"]["date_column"]].min(),
        subset_df[config["data"]["date_column"]].max(),
    )
    logger.info("Total unique dates in subset: %d", subset_df[config["data"]["date_column"]].nunique())

    subset_df.to_parquet(output_file, index=False)
    logger.info("Saved development subset to: %s", output_file)


if __name__ == "__main__":
    main()