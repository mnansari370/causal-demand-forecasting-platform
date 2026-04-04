from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_promotion_features(
    df: pd.DataFrame,
    promo_col: str,
    group_cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add promotion recency and promotion streak features per series.

    days_since_last_promo:
        Number of days since the series was last on promotion.
        NaN if no promotion has occurred yet.

    promo_streak:
        Number of consecutive days currently on promotion.
    """
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    def _days_since(series: pd.Series) -> pd.Series:
        result = np.full(len(series), np.nan)
        counter = np.nan
        for i, val in enumerate(series):
            if val == 1:
                counter = 0.0
            elif not np.isnan(counter):
                counter += 1.0
            result[i] = counter
        return pd.Series(result, index=series.index)

    def _streak(series: pd.Series) -> pd.Series:
        result = np.zeros(len(series))
        streak = 0
        for i, val in enumerate(series):
            streak = streak + 1 if val == 1 else 0
            result[i] = streak
        return pd.Series(result, index=series.index)

    df["days_since_last_promo"] = df.groupby(group_cols)[promo_col].transform(_days_since)
    df["promo_streak"] = df.groupby(group_cols)[promo_col].transform(_streak)

    logger.info("Added promotion features: days_since_last_promo, promo_streak")
    return df


def add_target_encoding(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str],
    train_mask: pd.Series,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """
    Smoothed target encoding using TRAIN rows only.

    encoded = (n * group_mean + k * global_mean) / (n + k)
    """
    global_mean = df.loc[train_mask, target_col].mean()
    logger.info("Target encoding global mean: %.4f", global_mean)

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning("Target encoding column '%s' not found, skipping", col)
            continue

        stats = df.loc[train_mask].groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )

        enc_col = f"{col}_target_enc"
        df[enc_col] = df[col].map(smoothed).fillna(global_mean)

        logger.info(
            "Target encoding created: %s -> %s (groups=%d)",
            col,
            enc_col,
            len(stats),
        )

    return df


def get_feature_columns(df: pd.DataFrame, config: dict) -> list[str]:
    """
    Single source of truth for model feature columns.
    Returns only columns that actually exist in df.
    """
    target_col = config["data"]["target_column"]
    promo_col = config["data"]["promo_column"]
    lag_days = config["features"]["lag_days"]
    windows = config["features"]["rolling_windows"]

    lag_cols = [f"{target_col}_lag_{d}" for d in lag_days]

    rolling_cols = []
    for w in windows:
        rolling_cols.extend(
            [
                f"{target_col}_rolling_mean_{w}d",
                f"{target_col}_rolling_std_{w}d",
            ]
        )

    calendar_cols = [
        "day_of_week",
        "is_weekend",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "days_to_year_end",
    ]

    promo_cols = [promo_col, "days_since_last_promo", "promo_streak"]
    external_cols = ["oil_price"]
    holiday_cols = ["is_holiday", "is_national_holiday", "is_local_holiday"]

    enc_cols = []
    if config["features"].get("use_target_encoding"):
        for col in config["features"].get("target_encoding_cols", []):
            enc_cols.append(f"{col}_target_enc")

    candidates = (
        lag_cols
        + rolling_cols
        + calendar_cols
        + promo_cols
        + external_cols
        + holiday_cols
        + enc_cols
    )

    available = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]

    if missing:
        logger.info("Feature columns not present and skipped: %s", missing)

    logger.info("Final feature matrix has %d columns", len(available))
    return available