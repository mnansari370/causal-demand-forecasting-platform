"""
Difference-in-Differences (DiD) implementation for promotion impact estimation.

DiD answers: "what was the true incremental demand lift from running promotions,
separate from seasonal trends and store-level differences?"

The key assumption is parallel trends: absent the promotion, treated and control
series would have evolved similarly. We validate this with a placebo test by
applying the same estimator to an earlier window where no real promotion effect
should appear. If the placebo estimate is close to zero, the design is more credible.

The ATT (Average Treatment Effect on the Treated) is the interaction term
coefficient in the regression:
    unit_sales_mean ~ treated + post + treated:post

The coefficient on treated:post is the causal estimate.
"""
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
    Build a DiD panel with one row per (store, item, period).

    Treatment is assigned based on whether a series had any promotion in the
    post-treatment window. Control series had no promotion in that same window.
    """
    pre_start_dt = pd.to_datetime(pre_start)
    pre_end_dt = pd.to_datetime(pre_end)
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    pre_df = df[(df[date_col] >= pre_start_dt) & (df[date_col] <= pre_end_dt)].copy()
    post_df = df[(df[date_col] >= post_start_dt) & (df[date_col] <= post_end_dt)].copy()

    logger.info(
        "DiD periods | pre: %s to %s (%d rows) | post: %s to %s (%d rows)",
        pre_start,
        pre_end,
        len(pre_df),
        post_start,
        post_end,
        len(post_df),
    )

    if pre_df.empty or post_df.empty:
        logger.error("One of the DiD periods is empty")
        return pd.DataFrame()

    group_cols = [store_col, item_col]

    post_promo = (
        post_df.groupby(group_cols)[treatment_col]
        .mean()
        .reset_index()
        .rename(columns={treatment_col: "promo_rate_post"})
    )
    post_promo["treated"] = (post_promo["promo_rate_post"] > 0).astype(int)

    n_treated = int(post_promo["treated"].sum())
    n_control = int((post_promo["treated"] == 0).sum())

    logger.info("Treated series: %d | Control series: %d", n_treated, n_control)

    if n_treated == 0 or n_control == 0:
        logger.error("No treated or no control series available for DiD")
        return pd.DataFrame()

    def aggregate_period(period_df: pd.DataFrame, post_flag: int) -> pd.DataFrame:
        agg = (
            period_df.groupby(group_cols)
            .agg(
                unit_sales_mean=(target_col, "mean"),
                n_rows=(target_col, "count"),
            )
            .reset_index()
        )

        min_required = min_post_rows if post_flag == 1 else min_pre_rows
        agg = agg[agg["n_rows"] >= min_required].copy()
        agg["post"] = post_flag

        agg = agg.merge(
            post_promo[group_cols + ["treated"]],
            on=group_cols,
            how="inner",
        )
        return agg

    panel = pd.concat(
        [
            aggregate_period(pre_df, post_flag=0),
            aggregate_period(post_df, post_flag=1),
        ],
        ignore_index=True,
    )

    panel = panel.dropna(subset=["unit_sales_mean", "treated", "post"]).reset_index(drop=True)

    logger.info(
        "DiD panel ready | rows=%d | treated series=%d | control series=%d",
        len(panel),
        panel.loc[panel["treated"] == 1, group_cols].drop_duplicates().shape[0],
        panel.loc[panel["treated"] == 0, group_cols].drop_duplicates().shape[0],
    )

    return panel


def run_did(panel: pd.DataFrame, label: str = "DiD") -> dict:
    """
    Run an OLS Difference-in-Differences regression.

    ATT is the coefficient on treated:post.
    """
    if len(panel) < 10:
        logger.warning("%s panel too small: %d rows", label, len(panel))
        return {}

    try:
        fit = smf.ols(
            "unit_sales_mean ~ treated + post + treated:post",
            data=panel,
        ).fit()
    except Exception as exc:
        logger.error("%s regression failed: %s", label, exc)
        return {}

    key = "treated:post"
    if key not in fit.params:
        logger.error("%s regression missing interaction term", label)
        return {}

    ci = fit.conf_int().loc[key]

    result = {
        "label": label,
        "estimate": round(float(fit.params[key]), 4),
        "std_error": round(float(fit.bse[key]), 4),
        "t_stat": round(float(fit.tvalues[key]), 4),
        "p_value": round(float(fit.pvalues[key]), 4),
        "ci_low": round(float(ci.iloc[0]), 4),
        "ci_high": round(float(ci.iloc[1]), 4),
        "n_obs": int(fit.nobs),
        "r_squared": round(float(fit.rsquared), 4),
        "significant": bool(fit.pvalues[key] < 0.05),
    }

    logger.info(
        "%s | ATT=%.4f | CI=[%.4f, %.4f] | p=%.4f | significant=%s",
        label,
        result["estimate"],
        result["ci_low"],
        result["ci_high"],
        result["p_value"],
        result["significant"],
    )

    return result


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
    Run DiD on an earlier placebo window.

    A small placebo estimate relative to the real estimate supports the
    credibility of the parallel-trends assumption.
    """
    logger.info(
        "Placebo test | fake pre: %s to %s | fake post: %s to %s",
        placebo_pre_start,
        placebo_pre_end,
        placebo_post_start,
        placebo_post_end,
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
        return {
            "label": "Placebo",
            "estimate": None,
            "passed": None,
            "verdict": "SKIPPED",
        }

    placebo_result = run_did(placebo_panel, label="Placebo")
    if not placebo_result:
        return {
            "label": "Placebo",
            "estimate": None,
            "passed": None,
            "verdict": "REGRESSION FAILED",
        }

    threshold = abs(real_did_estimate) / 2.0
    passed = abs(placebo_result["estimate"]) < threshold

    placebo_result["passed"] = passed
    placebo_result["threshold"] = round(threshold, 4)
    placebo_result["verdict"] = (
        "PASSED" if passed else "FAILED — possible parallel trends violation"
    )

    logger.info(
        "Placebo test | estimate=%.4f | threshold=%.4f | verdict=%s",
        placebo_result["estimate"],
        threshold,
        placebo_result["verdict"],
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
    Compare promoted vs not-promoted rows directly in the post period.

    This is intentionally naive and is used to quantify the selection bias
    that DiD is trying to correct.
    """
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    post_df = df[(df[date_col] >= post_start_dt) & (df[date_col] <= post_end_dt)].copy()

    promoted = post_df[post_df[treatment_col] == 1][target_col]
    not_promoted = post_df[post_df[treatment_col] == 0][target_col]

    if len(promoted) == 0 or len(not_promoted) == 0:
        logger.warning("Naive comparison could not be computed because one group is empty")
        return {}

    naive_estimate = float(promoted.mean() - not_promoted.mean())

    logger.info(
        "Naive comparison | promoted_mean=%.4f | not_promoted_mean=%.4f | estimate=%.4f",
        promoted.mean(),
        not_promoted.mean(),
        naive_estimate,
    )

    return {
        "naive_estimate": round(naive_estimate, 4),
        "promoted_mean_sales": round(float(promoted.mean()), 4),
        "unpromoted_mean_sales": round(float(not_promoted.mean()), 4),
        "n_promoted_rows": int(len(promoted)),
        "n_unpromoted_rows": int(len(not_promoted)),
    }