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

    date_col = config["data"]["date_column"]
    store_col = config["data"]["store_column"]

    total_days = (end_date - start_date).days + 1
    target_total_rows = rows_per_day * total_days

    logger.info("=" * 60)
    logger.info("Building forecasting-friendly development subset")
    logger.info("Input file: %s", train_file)
    logger.info("Date window: %s to %s", start_date.date(), end_date.date())
    logger.info("Rows/day target (old setting): %s", rows_per_day)
    logger.info("Total days: %s", total_days)
    logger.info("Approx target total rows: %s", target_total_rows)
    logger.info("Chunk size: %s", chunk_size)
    logger.info("Sampling strategy: fixed-store subset with full temporal continuity")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # PASS 1 — count rows per store within the requested date window
    # ---------------------------------------------------------------------
    logger.info("PASS 1: Counting rows per store in requested date window")

    store_counts: dict[int, int] = {}
    total_rows_seen = 0
    total_rows_in_window = 0

    reader = pd.read_csv(
        train_file,
        chunksize=chunk_size,
        parse_dates=[date_col],
        dtype={"onpromotion": "object"},
    )

    for i, chunk in enumerate(reader, start=1):
        total_rows_seen += len(chunk)

        chunk = chunk[
            (chunk[date_col] >= start_date) &
            (chunk[date_col] <= end_date)
        ]

        if chunk.empty:
            continue

        total_rows_in_window += len(chunk)

        counts = chunk[store_col].value_counts()
        for store, count in counts.items():
            store_counts[int(store)] = store_counts.get(int(store), 0) + int(count)

        if i % 20 == 0:
            logger.info(
                "PASS 1 progress | chunk=%d | total_seen=%d | rows_in_window=%d | stores_counted=%d",
                i,
                total_rows_seen,
                total_rows_in_window,
                len(store_counts),
            )

    if not store_counts:
        logger.error("No rows found in requested date range.")
        sys.exit(1)

    store_counts_series = pd.Series(store_counts).sort_values(ascending=False)

    logger.info("Rows in requested date window: %d", total_rows_in_window)
    logger.info("Unique stores available in window: %d", len(store_counts_series))
    logger.info("Top store row counts:\n%s", store_counts_series.head(20).to_string())

    # ---------------------------------------------------------------------
    # Choose a fixed set of stores whose cumulative row count is close to target
    # ---------------------------------------------------------------------
    selected_stores: list[int] = []
    cumulative_rows = 0

    for store, count in store_counts_series.items():
        if cumulative_rows < target_total_rows:
            selected_stores.append(int(store))
            cumulative_rows += int(count)
        else:
            break

    # Safety: ensure at least one store is selected
    if not selected_stores:
        selected_stores = [int(store_counts_series.index[0])]
        cumulative_rows = int(store_counts_series.iloc[0])

    logger.info("Selected %d stores", len(selected_stores))
    logger.info("Selected stores: %s", selected_stores)
    logger.info("Estimated subset rows from selected stores: %d", cumulative_rows)

    # ---------------------------------------------------------------------
    # PASS 2 — filter full rows for selected stores within the time window
    # ---------------------------------------------------------------------
    logger.info("PASS 2: Extracting subset rows for selected stores")

    selected_chunks = []
    total_rows_seen = 0
    kept_rows = 0

    reader = pd.read_csv(
        train_file,
        chunksize=chunk_size,
        parse_dates=[date_col],
        dtype={"onpromotion": "object"},
    )

    selected_store_set = set(selected_stores)

    for i, chunk in enumerate(reader, start=1):
        total_rows_seen += len(chunk)

        chunk = chunk[
            (chunk[date_col] >= start_date) &
            (chunk[date_col] <= end_date) &
            (chunk[store_col].isin(selected_store_set))
        ]

        if chunk.empty:
            continue

        selected_chunks.append(chunk)
        kept_rows += len(chunk)

        if i % 20 == 0 or len(chunk) > 0:
            logger.info(
                "PASS 2 progress | chunk=%d | total_seen=%d | kept_rows=%d",
                i,
                total_rows_seen,
                kept_rows,
            )

    if not selected_chunks:
        logger.error("No subset rows extracted for selected stores.")
        sys.exit(1)

    subset_df = pd.concat(selected_chunks, ignore_index=True)
    subset_df = subset_df.sort_values([store_col, date_col]).reset_index(drop=True)

    logger.info("=" * 60)
    logger.info("Final subset shape: %s", subset_df.shape)
    logger.info(
        "Subset date range: %s to %s",
        subset_df[date_col].min(),
        subset_df[date_col].max(),
    )
    logger.info("Total unique dates in subset: %d", subset_df[date_col].nunique())
    logger.info("Unique stores in subset: %d", subset_df[store_col].nunique())
    logger.info(
        "Rows per selected store:\n%s",
        subset_df[store_col].value_counts().sort_index().to_string(),
    )
    logger.info("=" * 60)

    subset_df.to_parquet(output_file, index=False)
    logger.info("Saved development subset to: %s", output_file)


if __name__ == "__main__":
    main()