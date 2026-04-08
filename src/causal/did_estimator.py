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
    pre_start_dt = pd.to_datetime(pre_start)
    pre_end_dt = pd.to_datetime(pre_end)
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    pre_df = df[(df[date_col] >= pre_start_dt) & (df[date_col] <= pre_end_dt)].copy()
    post_df = df[(df[date_col] >= post_start_dt) & (df[date_col] <= post_end_dt)].copy()

    logger.info(
        "DiD data — pre period: %s -> %s (%d rows) | post period: %s -> %s (%d rows)",
        pre_start, pre_end, len(pre_df),
        post_start, post_end, len(post_df),
    )

    group_cols = [store_col, item_col]

    post_promo = (
        post_df.groupby(group_cols)[treatment_col]
        .mean()
        .reset_index()
        .rename(columns={treatment_col: "promo_rate_post"})
    )
    post_promo["treated"] = (post_promo["promo_rate_post"] > 0).astype(int)

    pre_agg = (
        pre_df.groupby(group_cols)
        .agg(
            unit_sales_mean=(target_col, "mean"),
            n_rows=(target_col, "count"),
        )
        .reset_index()
    )
    pre_agg = pre_agg[pre_agg["n_rows"] >= min_pre_rows]
    pre_agg["post"] = 0

    post_agg = (
        post_df.groupby(group_cols)
        .agg(
            unit_sales_mean=(target_col, "mean"),
            n_rows=(target_col, "count"),
        )
        .reset_index()
    )
    post_agg = post_agg[post_agg["n_rows"] >= min_post_rows]
    post_agg["post"] = 1

    pre_agg = pre_agg.merge(post_promo[[*group_cols, "treated"]], on=group_cols, how="inner")
    post_agg = post_agg.merge(post_promo[[*group_cols, "treated"]], on=group_cols, how="inner")

    panel = pd.concat([pre_agg, post_agg], ignore_index=True)
    panel = panel.dropna(subset=["unit_sales_mean", "treated", "post"])

    treated_count = panel.loc[panel["treated"] == 1, group_cols].drop_duplicates().shape[0]
    control_count = panel.loc[panel["treated"] == 0, group_cols].drop_duplicates().shape[0]

    logger.info(
        "DiD panel: %d rows | treated series=%d | control series=%d",
        len(panel), treated_count, control_count,
    )

    return panel


def run_did(
    panel: pd.DataFrame,
    label: str = "DiD",
) -> dict:
    if len(panel) < 10:
        logger.warning("%s: insufficient panel rows (%d)", label, len(panel))
        return {}

    formula = "unit_sales_mean ~ treated + post + treated:post"

    try:
        result = smf.ols(formula, data=panel).fit()
    except Exception as exc:
        logger.error("%s OLS failed: %s", label, exc)
        return {}

    try:
        coef = result.params["treated:post"]
        se = result.bse["treated:post"]
        t_stat = result.tvalues["treated:post"]
        p_value = result.pvalues["treated:post"]
        ci = result.conf_int().loc["treated:post"]
    except KeyError:
        logger.error("%s: 'treated:post' not found in regression output", label)
        return {}

    output = {
        "label": label,
        "estimate": round(float(coef), 4),
        "std_error": round(float(se), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "ci_low": round(float(ci[0]), 4),
        "ci_high": round(float(ci[1]), 4),
        "n_obs": int(result.nobs),
        "r_squared": round(float(result.rsquared), 4),
        "significant": bool(p_value < 0.05),
    }

    logger.info(
        "%s | ATT=%.4f | SE=%.4f | t=%.4f | p=%.4f | CI=[%.4f, %.4f] | sig=%s",
        label,
        output["estimate"], output["std_error"],
        output["t_stat"], output["p_value"],
        output["ci_low"], output["ci_high"],
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
) -> dict:
    logger.info(
        "Running placebo test | pre: %s->%s | post: %s->%s",
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
    )

    if len(placebo_panel) < 10:
        logger.warning("Placebo panel too small (%d rows) — skipping", len(placebo_panel))
        return {"label": "Placebo", "estimate": None, "passed": None}

    placebo_result = run_did(placebo_panel, label="Placebo Test")

    if not placebo_result:
        return {"label": "Placebo", "estimate": None, "passed": None}

    threshold = abs(real_did_estimate) / 2 if real_did_estimate != 0 else 0.0
    passed = abs(placebo_result["estimate"]) < threshold if threshold > 0 else True
    placebo_result["passed"] = passed

    verdict = "PASSED" if passed else "FAILED — check parallel trends assumption"
    logger.info(
        "Placebo test %s | placebo_estimate=%.4f | real_estimate=%.4f",
        verdict, placebo_result["estimate"], real_did_estimate,
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
    post_start_dt = pd.to_datetime(post_start)
    post_end_dt = pd.to_datetime(post_end)

    post_df = df[(df[date_col] >= post_start_dt) & (df[date_col] <= post_end_dt)]

    promoted = post_df[post_df[treatment_col] == 1][target_col]
    not_promoted = post_df[post_df[treatment_col] == 0][target_col]

    naive_estimate = float(promoted.mean() - not_promoted.mean())

    logger.info(
        "Naive estimate | promoted_mean=%.4f | not_promoted_mean=%.4f | naive_lift=%.4f",
        promoted.mean(), not_promoted.mean(), naive_estimate,
    )

    return {
        "naive_estimate": round(naive_estimate, 4),
        "promoted_mean_sales": round(float(promoted.mean()), 4),
        "unpromoted_mean_sales": round(float(not_promoted.mean()), 4),
        "n_promoted_rows": int(len(promoted)),
        "n_unpromoted_rows": int(len(not_promoted)),
    }