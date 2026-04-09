"""
Build a temporally continuous development subset from the full Favorita train.csv.

Why this subset exists:
The full Favorita training file is very large, so for development we keep a
fixed date window and select a set of stores whose cumulative rows are close
to a target size. We keep all rows for those stores in the window so temporal
continuity is preserved.

This is important because lag and rolling features must be computed on complete
series, not on randomly sampled rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.utils.logger import get_logger


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "create_dev_subset",
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

    date_col = config["data"]["date_column"]
    store_col = config["data"]["store_column"]

    total_days = (end_date - start_date).days + 1
    target_total_rows = rows_per_day * total_days

    logger.info("Date window: %s to %s", start_date.date(), end_date.date())
    logger.info("Target total rows: ~%d", target_total_rows)
    logger.info("Chunk size: %d", chunk_size)

    logger.info("Pass 1: counting rows per store in the selected window")
    store_counts: dict[int, int] = {}

    reader = pd.read_csv(
        train_file,
        chunksize=chunk_size,
        parse_dates=[date_col],
        dtype={"onpromotion": "object"},
    )

    for chunk in reader:
        chunk = chunk[
            (chunk[date_col] >= start_date) &
            (chunk[date_col] <= end_date)
        ]

        if chunk.empty:
            continue

        counts = chunk[store_col].value_counts()
        for store, count in counts.items():
            store_counts[int(store)] = store_counts.get(int(store), 0) + int(count)

    if not store_counts:
        logger.error("No rows found in the requested date window")
        sys.exit(1)

    store_counts_series = pd.Series(store_counts).sort_values(ascending=False)

    selected_stores: list[int] = []
    cumulative_rows = 0

    for store, count in store_counts_series.items():
        if cumulative_rows >= target_total_rows:
            break
        selected_stores.append(int(store))
        cumulative_rows += int(count)

    if not selected_stores:
        selected_stores = [int(store_counts_series.index[0])]
        cumulative_rows = int(store_counts_series.iloc[0])

    logger.info("Selected %d stores", len(selected_stores))
    logger.info("Estimated rows in subset: %d", cumulative_rows)

    logger.info("Pass 2: extracting rows for selected stores")
    selected_store_set = set(selected_stores)
    selected_chunks = []

    reader = pd.read_csv(
        train_file,
        chunksize=chunk_size,
        parse_dates=[date_col],
        dtype={"onpromotion": "object"},
    )

    for chunk in reader:
        chunk = chunk[
            (chunk[date_col] >= start_date) &
            (chunk[date_col] <= end_date) &
            (chunk[store_col].isin(selected_store_set))
        ]

        if not chunk.empty:
            selected_chunks.append(chunk)

    if not selected_chunks:
        logger.error("No rows extracted for selected stores")
        sys.exit(1)

    subset_df = (
        pd.concat(selected_chunks, ignore_index=True)
        .sort_values([store_col, date_col])
        .reset_index(drop=True)
    )

    logger.info("Final subset shape: %s", subset_df.shape)
    logger.info(
        "Subset date range: %s to %s",
        subset_df[date_col].min(),
        subset_df[date_col].max(),
    )
    logger.info("Unique stores in subset: %d", subset_df[store_col].nunique())

    subset_df.to_parquet(output_file, index=False)
    logger.info("Saved development subset to: %s", output_file)


if __name__ == "__main__":
    main()