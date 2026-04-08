from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_did_data(
    df: pd.DataFrame,
    treatment_col: str,
    target_col: str,
    date_col: str,
    store_col: str,
    item_col: str,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    min_pre_rows: int = 4,
    min_post_rows: int = 4,
) -> pd.DataFrame:
    """
    Build a long-format DiD panel with one row per (store, item, period).

    A series is treated if it has any promotion activity in the post period.
    """
    pre_start_dt = pd.to_datetime(pre_start)
    pre_end_dt = pd.to_datetime(pre_end)
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    pre_df = df[(df[date_col] >= pre_start_dt) & (df[date_col] <= pre_end_dt)].copy()
    post_df = df[(df[date_col] >= post_start_dt) & (df[date_col] <= post_end_dt)].copy()

    logger.info(
        "DiD periods — pre: %s to %s (%d rows) | post: %s to %s (%d rows)",
        pre_start, pre_end, len(pre_df),
        post_start, post_end, len(post_df),
    )

    if len(pre_df) == 0 or len(post_df) == 0:
        logger.error("One of the DiD periods is empty. Check the date ranges.")
        return pd.DataFrame()

    group_cols = [store_col, item_col]

    # Treatment assignment from post period
    post_promo = (
        post_df.groupby(group_cols)[treatment_col]
        .mean()
        .reset_index()
        .rename(columns={treatment_col: "promo_rate_post"})
    )
    post_promo["treated"] = (post_promo["promo_rate_post"] > 0).astype(int)

    n_treated = int(post_promo["treated"].sum())
    n_control = int((post_promo["treated"] == 0).sum())
    logger.info(
        "Treatment assignment — treated series: %d | control series: %d",
        n_treated, n_control,
    )

    if n_treated == 0 or n_control == 0:
        logger.error("No treated or no control series found in post period.")
        return pd.DataFrame()

    # Pre-period aggregates
    pre_agg = (
        pre_df.groupby(group_cols)
        .agg(
            unit_sales_mean=(target_col, "mean"),
            n_rows=(target_col, "count"),
        )
        .reset_index()
    )
    pre_agg = pre_agg[pre_agg["n_rows"] >= min_pre_rows].copy()
    pre_agg["post"] = 0

    # Post-period aggregates
    post_agg = (
        post_df.groupby(group_cols)
        .agg(
            unit_sales_mean=(target_col, "mean"),
            n_rows=(target_col, "count"),
        )
        .reset_index()
    )
    post_agg = post_agg[post_agg["n_rows"] >= min_post_rows].copy()
    post_agg["post"] = 1

    pre_agg = pre_agg.merge(post_promo[group_cols + ["treated"]], on=group_cols, how="inner")
    post_agg = post_agg.merge(post_promo[group_cols + ["treated"]], on=group_cols, how="inner")

    panel = pd.concat([pre_agg, post_agg], ignore_index=True)
    panel = panel.dropna(subset=["unit_sales_mean", "treated", "post"])

    treated_series = panel.loc[panel["treated"] == 1, group_cols].drop_duplicates().shape[0]
    control_series = panel.loc[panel["treated"] == 0, group_cols].drop_duplicates().shape[0]

    logger.info(
        "DiD panel ready — %d rows | %d treated series | %d control series",
        len(panel), treated_series, control_series,
    )

    return panel


def run_did(
    panel: pd.DataFrame,
    label: str = "DiD",
) -> dict:
    """
    OLS DiD:
      unit_sales_mean ~ treated + post + treated:post

    ATT is the coefficient on treated:post
    """
    if len(panel) < 10:
        logger.warning("%s: panel too small (%d rows)", label, len(panel))
        return {}

    formula = "unit_sales_mean ~ treated + post + treated:post"

    try:
        model = smf.ols(formula, data=panel)
        result = model.fit()
    except Exception as exc:
        logger.error("%s regression failed: %s", label, exc)
        return {}

    interaction_key = "treated:post"
    if interaction_key not in result.params:
        logger.error("%s: interaction term '%s' missing", label, interaction_key)
        return {}

    coef = result.params[interaction_key]
    se = result.bse[interaction_key]
    t_stat = result.tvalues[interaction_key]
    p_value = result.pvalues[interaction_key]
    ci = result.conf_int().loc[interaction_key]

    output = {
        "label": label,
        "estimate": round(float(coef), 4),
        "std_error": round(float(se), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "ci_low": round(float(ci.iloc[0]), 4),
        "ci_high": round(float(ci.iloc[1]), 4),
        "n_obs": int(result.nobs),
        "r_squared": round(float(result.rsquared), 4),
        "significant": bool(p_value < 0.05),
    }

    logger.info(
        "%s ATT=%.4f [%.4f, %.4f] | p=%.4f | significant=%s",
        label,
        output["estimate"],
        output["ci_low"],
        output["ci_high"],
        output["p_value"],
        output["significant"],
    )

    return output


def run_placebo_test(
    df: pd.DataFrame,
    treatment_col: str,
    target_col: str,
    date_col: str,
    store_col: str,
    item_col: str,
    placebo_pre_start: str,
    placebo_pre_end: str,
    placebo_post_start: str,
    placebo_post_end: str,
    real_did_estimate: float,
    min_pre_rows: int = 4,
    min_post_rows: int = 4,
) -> dict:
    """
    Run fake DiD on an earlier window where no real effect should appear.
    """
    logger.info(
        "Running placebo test — fake pre: %s→%s | fake post: %s→%s",
        placebo_pre_start, placebo_pre_end,
        placebo_post_start, placebo_post_end,
    )

    placebo_panel = prepare_did_data(
        df=df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        pre_start=placebo_pre_start,
        pre_end=placebo_pre_end,
        post_start=placebo_post_start,
        post_end=placebo_post_end,
        min_pre_rows=min_pre_rows,
        min_post_rows=min_post_rows,
    )

    if len(placebo_panel) < 10:
        logger.warning("Placebo panel too small (%d rows)", len(placebo_panel))
        return {
            "label": "Placebo Test",
            "estimate": None,
            "passed": None,
            "verdict": "SKIPPED — insufficient data",
        }

    placebo_result = run_did(placebo_panel, label="Placebo Test")

    if not placebo_result:
        return {
            "label": "Placebo Test",
            "estimate": None,
            "passed": None,
            "verdict": "FAILED — regression error",
        }

    threshold = abs(real_did_estimate) / 2.0
    passed = abs(placebo_result["estimate"]) < threshold

    placebo_result["passed"] = passed
    placebo_result["verdict"] = "PASSED" if passed else "FAILED — parallel trends violation"
    placebo_result["threshold"] = round(threshold, 4)

    logger.info(
        "Placebo %s | placebo_estimate=%.4f | threshold=%.4f | real_estimate=%.4f",
        placebo_result["verdict"],
        placebo_result["estimate"],
        threshold,
        real_did_estimate,
    )

    return placebo_result


def naive_vs_did_comparison(
    df: pd.DataFrame,
    treatment_col: str,
    target_col: str,
    store_col: str,
    item_col: str,
    date_col: str,
    post_start: str,
    post_end: str,
) -> dict:
    """
    Simple promoted vs non-promoted comparison in post window.
    This is biased and used only as a benchmark against DiD.
    """
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    post_df = df[
        (df[date_col] >= post_start_dt) &
        (df[date_col] <= post_end_dt)
    ].copy()

    promoted = post_df[post_df[treatment_col] == 1][target_col]
    not_promoted = post_df[post_df[treatment_col] == 0][target_col]

    if len(promoted) == 0 or len(not_promoted) == 0:
        logger.warning("Naive comparison failed: one group is empty.")
        return {}

    naive_estimate = float(promoted.mean() - not_promoted.mean())

    result = {
        "naive_estimate": round(naive_estimate, 4),
        "promoted_mean_sales": round(float(promoted.mean()), 4),
        "unpromoted_mean_sales": round(float(not_promoted.mean()), 4),
        "n_promoted_rows": int(len(promoted)),
        "n_unpromoted_rows": int(len(not_promoted)),
    }

    logger.info(
        "Naive estimate=%.4f | promoted_mean=%.4f | unpromoted_mean=%.4f",
        naive_estimate,
        result["promoted_mean_sales"],
        result["unpromoted_mean_sales"],
    )

    return result