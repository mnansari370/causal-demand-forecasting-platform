"""
Data loading utilities for the Favorita dataset.

Favorita is a multi-table retail dataset. The main train.csv is very large,
so during development we often work with a date-filtered store subset saved
as a parquet file. This module handles both cases through the config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_safe(
    file_path: Path,
    parse_dates: list[str] | None = None,
    nrows: int | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame | None:
    """
    Load a CSV safely.

    Returns None if the file does not exist instead of raising immediately.
    This is useful for optional side tables.
    """
    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return None

    logger.info("Loading: %s", file_path)
    df = pd.read_csv(file_path, parse_dates=parse_dates, nrows=nrows, dtype=dtype)
    logger.info("Loaded %s — shape: %s", file_path.name, df.shape)
    return df


def load_favorita_data(
    config: dict,
    sample_rows: int | None = None,
) -> Dict[str, pd.DataFrame | None]:
    """
    Load Favorita tables.

    If dev_subset.enabled is true and the subset parquet exists, it is used
    instead of reading the full raw train.csv. This keeps experimentation fast
    while preserving temporal continuity inside selected stores.
    """
    raw_dir = Path(config["data"]["raw_data_dir"])
    interim_dir = Path(config["data"]["interim_data_dir"])
    files = config["data"]["files"]
    date_col = config["data"]["date_column"]

    dev_cfg = config.get("dev_subset", {})
    dev_subset_enabled = dev_cfg.get("enabled", False)
    subset_path = interim_dir / dev_cfg.get("output_file", "train_dev_subset.parquet")

    n_rows = sample_rows if sample_rows is not None else config["preprocessing"].get("sample_rows")

    if dev_subset_enabled and subset_path.exists():
        logger.info("Loading development subset from: %s", subset_path)
        train_df = pd.read_parquet(subset_path)
    else:
        train_df = load_csv_safe(
            raw_dir / files["train"],
            parse_dates=[date_col],
            nrows=n_rows,
            dtype={"onpromotion": "object"},
        )

    data: Dict[str, pd.DataFrame | None] = {
        "train": train_df,
        "test": load_csv_safe(raw_dir / files["test"], parse_dates=[date_col]),
        "stores": load_csv_safe(raw_dir / files["stores"]),
        "items": load_csv_safe(raw_dir / files["items"]),
        "transactions": load_csv_safe(raw_dir / files["transactions"], parse_dates=[date_col]),
        "oil": load_csv_safe(raw_dir / files["oil"], parse_dates=[date_col]),
        "holidays_events": load_csv_safe(raw_dir / files["holidays_events"], parse_dates=[date_col]),
    }

    return data


def summarize_dataframe(name: str, df: pd.DataFrame | None) -> None:
    """
    Log a compact summary of a dataframe.
    """
    if df is None:
        logger.warning("%s: not loaded", name)
        return

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    logger.info("%s shape: %s", name, df.shape)
    logger.info("%s columns: %s", name, list(df.columns))

    if not missing.empty:
        logger.info("%s missing values:\n%s", name, missing.to_string())

    logger.info("%s head:\n%s", name, df.head(3).to_string())