from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_train(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Basic cleaning of training data:
    - fill missing promotions
    - clip negative sales
    - optional log transform
    - filter date range
    """
    df = df.copy()

    target_col = config["data"]["target_column"]
    promo_col = config["data"]["promo_column"]
    date_col = config["data"]["date_column"]
    prep = config["preprocessing"]

    if prep.get("fill_missing_promotions", False) and promo_col in df.columns:
        df[promo_col] = df[promo_col].fillna(False)
        df[promo_col] = df[promo_col].map(lambda x: True if str(x).strip().lower() == "true" else False)
        df[promo_col] = df[promo_col].astype(int)

    if prep.get("clip_negative_sales", False) and target_col in df.columns:
        n_neg = (df[target_col] < 0).sum()
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
    if stores is None:
        logger.warning("stores table missing; skipping merge")
        return df
    return df.merge(stores, on="store_nbr", how="left")


def merge_items(df: pd.DataFrame, items: pd.DataFrame | None) -> pd.DataFrame:
    if items is None:
        logger.warning("items table missing; skipping merge")
        return df
    return df.merge(items, on="item_nbr", how="left")


def merge_oil(df: pd.DataFrame, oil: pd.DataFrame | None, date_col: str = "date") -> pd.DataFrame:
    if oil is None:
        logger.warning("oil table missing; skipping merge")
        return df

    oil = oil.copy().rename(columns={"dcoilwtico": "oil_price"})

    full_dates = pd.DataFrame({
        date_col: pd.date_range(oil[date_col].min(), oil[date_col].max())
    })

    oil = full_dates.merge(oil, on=date_col, how="left")
    oil["oil_price"] = oil["oil_price"].ffill().bfill()

    return df.merge(oil[[date_col, "oil_price"]], on=date_col, how="left")


def merge_holidays(df: pd.DataFrame, holidays: pd.DataFrame | None, date_col: str = "date") -> pd.DataFrame:
    if holidays is None:
        logger.warning("holidays_events table missing; skipping merge")
        return df

    h = holidays.copy()
    h["is_holiday"] = 1

    national = h[h["locale"] == "National"][[date_col]].drop_duplicates()
    national["is_national_holiday"] = 1

    local = h[h["locale"] != "National"][[date_col]].drop_duplicates()
    local["is_local_holiday"] = 1

    df = df.merge(
        h[[date_col, "is_holiday"]].drop_duplicates().groupby(date_col).max().reset_index(),
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
    d = df[date_col]

    df["day_of_week"] = d.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = d.dt.day
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["days_to_year_end"] = (pd.to_datetime(d.dt.year.astype(str) + "-12-31") - d).dt.days

    return df


def temporal_split(df: pd.DataFrame, date_col: str, splits: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[date_col] <= pd.to_datetime(splits["train_end"])]
    val = df[
        (df[date_col] >= pd.to_datetime(splits["val_start"])) &
        (df[date_col] <= pd.to_datetime(splits["val_end"]))
    ]
    test = df[
        (df[date_col] >= pd.to_datetime(splits["test_start"])) &
        (df[date_col] <= pd.to_datetime(splits["test_end"]))
    ]

    logger.info("Temporal split sizes -> train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def save_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved parquet: %s", output_path)


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
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
    Add lag features per group.
    Example: yesterday's sales, last week's sales, etc.
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
    Add rolling mean and rolling std features based on past values only.
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