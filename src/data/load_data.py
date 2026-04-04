from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str | Path) -> dict:
    """
    Load YAML configuration.
    Single shared config loader for the whole project.
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
    Load a CSV safely. Returns None if file does not exist.
    """
    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return None

    logger.info("Loading file: %s", file_path)
    df = pd.read_csv(file_path, parse_dates=parse_dates, nrows=nrows, dtype=dtype)
    logger.info("Loaded shape for %s: %s", file_path.name, df.shape)
    return df


def load_favorita_data(config: dict, sample_rows: int | None = None) -> Dict[str, pd.DataFrame | None]:
    """
    Load Favorita tables.
    If dev_subset.enabled is true and the subset file exists, load that instead of raw train.csv.
    """
    raw_dir = Path(config["data"]["raw_data_dir"])
    interim_dir = Path(config["data"]["interim_data_dir"])
    files = config["data"]["files"]
    date_col = config["data"]["date_column"]

    dev_subset_cfg = config.get("dev_subset", {})
    dev_subset_enabled = dev_subset_cfg.get("enabled", False)
    dev_subset_file = interim_dir / dev_subset_cfg.get("output_file", "train_dev_subset.parquet")

    n_rows = sample_rows if sample_rows is not None else config["preprocessing"].get("sample_rows")

    if dev_subset_enabled and dev_subset_file.exists():
        logger.info("Loading development subset from: %s", dev_subset_file)
        train_df = pd.read_parquet(dev_subset_file)
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
    Log compact summary of a dataframe.
    """
    if df is None:
        logger.warning("%s: NOT LOADED", name)
        return

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    logger.info("%s shape: %s", name, df.shape)
    logger.info("%s columns: %s", name, list(df.columns))

    if not missing.empty:
        logger.info("%s missing values:\n%s", name, missing.to_string())

    logger.info("%s head:\n%s", name, df.head(3).to_string())