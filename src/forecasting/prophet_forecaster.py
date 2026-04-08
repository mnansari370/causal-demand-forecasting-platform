from __future__ import annotations

import warnings

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def run_prophet_on_sample(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    n_series: int = 10,
) -> pd.DataFrame:
    """
    Run Prophet on the top-N series by total training volume.

    Important:
    The test period is not immediately after the train period in your setup.
    Train ends on 2016-12-31, while test starts on 2017-07-01.
    So Prophet must forecast far enough ahead to cover the full gap.

    Returns columns:
      store_nbr, item_nbr, date, unit_sales, yhat, yhat_lower, yhat_upper
    """
    from prophet import Prophet

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

    logger.info("Prophet: running on top %d series", len(top_series))

    all_results = []

    for idx, (store, item) in enumerate(top_series, start=1):
        logger.info("Prophet [%d/%d] store=%s item=%s", idx, len(top_series), store, item)

        train_s = train_df[
            (train_df[store_col] == store) & (train_df[item_col] == item)
        ][[date_col, target_col]].sort_values(date_col).copy()

        test_s = test_df[
            (test_df[store_col] == store) & (test_df[item_col] == item)
        ][[date_col, target_col]].sort_values(date_col).copy()

        logger.info("train_s rows=%d | test_s rows=%d", len(train_s), len(test_s))

        if len(train_s) < 30 or len(test_s) == 0:
            logger.info("Skipping Prophet series: insufficient rows")
            continue

        train_s[date_col] = pd.to_datetime(train_s[date_col])
        test_s[date_col] = pd.to_datetime(test_s[date_col])

        prophet_df = train_s.rename(columns={date_col: "ds", target_col: "y"})
        prophet_df["y"] = prophet_df["y"].clip(lower=0)

        train_max_date = prophet_df["ds"].max()
        test_max_date = test_s[date_col].max()

        horizon_days = (test_max_date - train_max_date).days
        logger.info(
            "Prophet horizon | train_max=%s | test_max=%s | periods=%d",
            train_max_date.date(),
            test_max_date.date(),
            horizon_days,
        )

        if horizon_days <= 0:
            logger.warning(
                "Skipping Prophet series: non-positive horizon for store=%s item=%s",
                store,
                item,
            )
            continue

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.90,
                changepoint_prior_scale=0.05,
            )

            # Keep holidays disabled while debugging robustness
            # model.add_country_holidays(country_name="EC")

            model.fit(prophet_df)

            # Forecast all the way from train end to test end
            future = model.make_future_dataframe(periods=horizon_days, freq="D")
            forecast = model.predict(future)

            forecast["ds"] = pd.to_datetime(forecast["ds"])

            # Keep only rows that belong to the actual test dates
            fc = forecast.merge(
                test_s[[date_col]].rename(columns={date_col: "ds"}),
                on="ds",
                how="inner",
            )[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

            logger.info("matched test forecast rows=%d", len(fc))

            if fc.empty:
                logger.warning("Prophet matched zero rows for store=%s item=%s", store, item)
                continue

            fc = fc.rename(columns={"ds": date_col}).reset_index(drop=True)
            fc["yhat"] = fc["yhat"].clip(lower=0)
            fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)

            actual_map = test_s.set_index(date_col)[target_col]
            fc[target_col] = fc[date_col].map(actual_map).values
            fc[store_col] = store
            fc[item_col] = item

            all_results.append(fc)

        except Exception as exc:
            logger.warning("Prophet failed for store=%s item=%s | %s", store, item, exc)
            continue

    if not all_results:
        logger.warning("Prophet produced no results")
        return pd.DataFrame()

    result_df = pd.concat(all_results, ignore_index=True)
    logger.info("Prophet complete: %d rows across %d series", len(result_df), len(all_results))
    return result_df