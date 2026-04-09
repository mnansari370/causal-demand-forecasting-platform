"""
SARIMAX forecaster run on a sample of high-volume series.

SARIMAX is included as a classical statistical benchmark. It models each
series separately and can use a small set of exogenous variables such as
promotion status, oil price, and holidays.
"""
from __future__ import annotations

import warnings

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def run_sarimax_on_sample(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    n_series: int = 20,
) -> pd.DataFrame:
    """
    Run SARIMAX on the top-N series by training demand volume.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]

    top_series = (
        train_df.groupby([store_col, item_col])[target_col]
        .sum()
        .sort_values(ascending=False)
        .head(n_series)
        .index.tolist()
    )

    exog_candidates = ["onpromotion", "oil_price", "is_holiday"]
    exog_cols = [c for c in exog_candidates if c in train_df.columns]

    logger.info("SARIMAX: %d series | exog=%s", len(top_series), exog_cols)

    all_results = []

    for idx, (store, item) in enumerate(top_series, start=1):
        logger.info("SARIMAX [%d/%d] | store=%s item=%s", idx, len(top_series), store, item)

        mask_tr = (train_df[store_col] == store) & (train_df[item_col] == item)
        mask_te = (test_df[store_col] == store) & (test_df[item_col] == item)

        train_s = train_df[mask_tr].set_index(date_col)[target_col].sort_index()
        test_s = test_df[mask_te].set_index(date_col)[target_col].sort_index()

        if len(train_s) < 90 or len(test_s) == 0:
            logger.info("Skipping SARIMAX series: train=%d test=%d", len(train_s), len(test_s))
            continue

        exog_train = (
            train_df[mask_tr].set_index(date_col)[exog_cols].sort_index()
            if exog_cols else None
        )
        exog_test = (
            test_df[mask_te].set_index(date_col)[exog_cols].sort_index()
            if exog_cols else None
        )

        try:
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 7) if len(train_s) >= 180 else (0, 0, 0, 0)

            model = SARIMAX(
                train_s,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=300)

            forecast_obj = fit.get_forecast(steps=len(test_s), exog=exog_test)
            ci = forecast_obj.conf_int(alpha=0.10)

            row = pd.DataFrame({
                date_col: test_s.index,
                target_col: test_s.values,
                "forecast": forecast_obj.predicted_mean.values.clip(min=0),
                "lower_ci": ci.iloc[:, 0].values.clip(min=0),
                "upper_ci": ci.iloc[:, 1].values.clip(min=0),
                store_col: store,
                item_col: item,
            })

            all_results.append(row)

        except Exception as exc:
            logger.warning("SARIMAX failed for store=%s item=%s | %s", store, item, exc)

    if not all_results:
        logger.warning("SARIMAX produced no results")
        return pd.DataFrame()

    result_df = pd.concat(all_results, ignore_index=True)
    logger.info("SARIMAX complete: %d rows across %d series", len(result_df), len(all_results))
    return result_df