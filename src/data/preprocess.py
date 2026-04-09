"""
Data cleaning, merging, and temporal splitting for Favorita.

Design notes:
- Splits are always chronological. We never shuffle time-series data.
- The raw 'onpromotion' column comes in as mixed object/string values and
  needs explicit normalisation before it can be used as a model feature.
- Oil price has missing dates, so we forward-fill and back-fill after
  expanding to a complete daily date range.
- Holiday information is split into national and local signals because
  they affect demand differently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_train(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply basic cleaning to the training data.

    Steps:
    - normalise promotion values to 0/1
    - clip negative sales to zero
    - optionally apply log1p to the target
    - restrict rows to the configured date window
    """
    df = df.copy()

    target_col = config["data"]["target_column"]
    promo_col = config["data"]["promo_column"]
    date_col = config["data"]["date_column"]
    prep = config["preprocessing"]

    if promo_col in df.columns:
        logger.info("Promotion dtype before cleaning: %s", df[promo_col].dtype)

    if prep.get("fill_missing_promotions", False) and promo_col in df.columns:
        # Raw CSV values are mixed booleans / strings / NaN, so we standardise
        # them into an integer 0/1 representation.
        df[promo_col] = df[promo_col].fillna(False)
        df[promo_col] = df[promo_col].map(
            lambda x: 1 if str(x).strip().lower() == "true" else 0
        )
        df[promo_col] = df[promo_col].astype(int)

        logger.info(
            "Promotion value counts after cleaning: %s",
            df[promo_col].value_counts(dropna=False).to_dict(),
        )

    if prep.get("clip_negative_sales", False) and target_col in df.columns:
        n_neg = int((df[target_col] < 0).sum())
        if n_neg > 0:
            logger.info("Clipping %d negative sales values to 0", n_neg)
        df[target_col] = df[target_col].clip(lower=0)

    if prep.get("log_transform_target", False) and target_col in df.columns:
        logger.info("Applying log1p transform to %s", target_col)
        df[target_col] = np.log1p(df[target_col])

    min_date = prep.get("min_date")
    max_date = prep.get("max_date")

    if min_date:
        df = df[df[date_col] >= pd.to_datetime(min_date)]
    if max_date:
        df = df[df[date_col] <= pd.to_datetime(max_date)]

    return df.reset_index(drop=True)


def merge_stores(df: pd.DataFrame, stores: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge store metadata into the main dataframe.
    """
    if stores is None:
        logger.warning("Stores table missing — skipping merge")
        return df
    return df.merge(stores, on="store_nbr", how="left")


def merge_items(df: pd.DataFrame, items: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge item metadata into the main dataframe.
    """
    if items is None:
        logger.warning("Items table missing — skipping merge")
        return df
    return df.merge(items, on="item_nbr", how="left")


def merge_oil(
    df: pd.DataFrame,
    oil: pd.DataFrame | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Merge daily oil price information.

    Missing oil-price days are filled by carrying the last known value
    forward, then backward for any initial gap.
    """
    if oil is None:
        logger.warning("Oil table missing — skipping merge")
        return df

    oil = oil.copy().rename(columns={"dcoilwtico": "oil_price"})

    full_dates = pd.DataFrame({
        date_col: pd.date_range(oil[date_col].min(), oil[date_col].max())
    })

    oil = full_dates.merge(oil, on=date_col, how="left")
    oil["oil_price"] = oil["oil_price"].ffill().bfill()

    return df.merge(oil[[date_col, "oil_price"]], on=date_col, how="left")


def merge_holidays(
    df: pd.DataFrame,
    holidays: pd.DataFrame | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Merge holiday signals.

    We keep:
    - is_holiday
    - is_national_holiday
    - is_local_holiday

    This is intentionally simple and avoids overfitting to holiday labels.
    """
    if holidays is None:
        logger.warning("Holidays table missing — skipping merge")
        return df

    h = holidays.copy()
    h["is_holiday"] = 1

    national = h[h["locale"] == "National"][[date_col]].drop_duplicates()
    national["is_national_holiday"] = 1

    local = h[h["locale"] != "National"][[date_col]].drop_duplicates()
    local["is_local_holiday"] = 1

    df = df.merge(
        h[[date_col, "is_holiday"]]
        .drop_duplicates()
        .groupby(date_col)
        .max()
        .reset_index(),
        on=date_col,
        how="left",
    )

    df = df.merge(national, on=date_col, how="left")
    df = df.merge(local, on=date_col, how="left")

    for col in ["is_holiday", "is_national_holiday", "is_local_holiday"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar-based predictors.

    These features are cheap to compute and typically very useful in retail
    demand forecasting, especially day-of-week and month effects.
    """
    d = df[date_col]

    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = d.dt.day
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["days_to_year_end"] = (
        pd.to_datetime(d.dt.year.astype(str) + "-12-31") - d
    ).dt.days

    return df


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    splits: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test using time boundaries.

    This is the correct split strategy for forecasting tasks because it
    prevents future information from leaking into training.
    """
    train = df[df[date_col] <= pd.to_datetime(splits["train_end"])]
    val = df[
        (df[date_col] >= pd.to_datetime(splits["val_start"])) &
        (df[date_col] <= pd.to_datetime(splits["val_end"]))
    ]
    test = df[
        (df[date_col] >= pd.to_datetime(splits["test_start"])) &
        (df[date_col] <= pd.to_datetime(splits["test_end"]))
    ]

    logger.info(
        "Temporal split | train=%d val=%d test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


def save_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a dataframe as parquet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved parquet: %s", output_path)


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a dataframe as CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved CSV: %s", output_path)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    lag_days: list[int],
) -> pd.DataFrame:
    """
    Add lag features within each series.

    Example:
    - lag 1  -> yesterday's sales
    - lag 7  -> same weekday last week
    - lag 28 -> same period last month
    """
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    for lag in lag_days:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        logger.info("Added lag feature: %s", col_name)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """
    Add rolling mean and rolling std based on past values only.

    The shift(1) is important: it ensures the current day's target is not
    used to build its own features.
    """
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    for window in windows:
        mean_col = f"{target_col}_rolling_mean_{window}d"
        std_col = f"{target_col}_rolling_std_{window}d"

        grouped = df.groupby(group_cols)[target_col]

        df[mean_col] = grouped.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[std_col] = grouped.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

        logger.info("Added rolling features: %s, %s", mean_col, std_col)

    return df