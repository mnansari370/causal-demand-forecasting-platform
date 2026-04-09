"""
Promotion sensitivity estimation.

Important terminology note:
We use the term "promotion sensitivity" rather than strict "price elasticity"
because the Favorita dataset does not contain actual selling prices. Instead,
we estimate how demand changes when an item is on promotion.

Two models are used:
1. Family-level OLS
2. Panel OLS with item fixed effects
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def estimate_family_promotion_sensitivity(
    df: pd.DataFrame,
    target_col: str,
    promo_col: str,
    family_col: str,
    control_cols: list[str],
    min_obs: int = 100,
) -> pd.DataFrame:
    """
    Estimate promotion sensitivity separately for each product family.

    Model:
        log1p(demand) ~ promotion + controls
    """
    if family_col not in df.columns:
        logger.warning("Family column '%s' not found", family_col)
        return pd.DataFrame()

    available_controls = [c for c in control_cols if c in df.columns]
    families = sorted(df[family_col].dropna().unique().tolist())

    logger.info("Estimating family-level promotion sensitivity for %d families", len(families))

    results = []

    for family in families:
        sub = df[df[family_col] == family].copy()
        sub = sub.dropna(subset=[target_col, promo_col])

        if len(sub) < min_obs:
            continue

        sub["log_demand"] = np.log1p(sub[target_col].clip(lower=0))

        rhs_terms = [promo_col] + available_controls
        formula = "log_demand ~ " + " + ".join(rhs_terms)

        try:
            fit = smf.ols(formula, data=sub).fit()
        except Exception as exc:
            logger.warning("Family regression failed for '%s': %s", family, exc)
            continue

        if promo_col not in fit.params:
            continue

        coef = float(fit.params[promo_col])
        ci = fit.conf_int().loc[promo_col]
        p_value = float(fit.pvalues[promo_col])

        results.append(
            {
                "family": family,
                "promotion_coef": round(coef, 4),
                "pct_demand_change": round((np.exp(coef) - 1.0) * 100.0, 2),
                "std_error": round(float(fit.bse[promo_col]), 4),
                "p_value": round(p_value, 4),
                "ci_low": round(float(ci.iloc[0]), 4),
                "ci_high": round(float(ci.iloc[1]), 4),
                "n_obs": int(len(sub)),
                "significant": bool(p_value < 0.05),
                "r_squared": round(float(fit.rsquared), 4),
            }
        )

    if not results:
        logger.warning("No family-level promotion sensitivity results produced")
        return pd.DataFrame()

    result_df = (
        pd.DataFrame(results)
        .sort_values("promotion_coef", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Family sensitivity complete | families=%d | significant=%d",
        len(result_df),
        int(result_df["significant"].sum()),
    )
    return result_df


def estimate_panel_promotion_sensitivity(
    df: pd.DataFrame,
    target_col: str,
    promo_col: str,
    date_col: str,
    item_col: str,
    min_rows: int = 500,
    max_items: int = 200,
) -> dict:
    """
    Estimate average promotion effect using panel regression with item fixed effects.

    This compares each item against itself across time, which is stronger than
    comparing different items to one another.
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        logger.warning("linearmodels is not installed")
        return {}

    sub = df.dropna(subset=[target_col, promo_col, item_col, date_col]).copy()
    sub = sub[sub[target_col] >= 0].copy()

    item_counts = sub.groupby(item_col)[target_col].count().sort_values(ascending=False)
    valid_items = item_counts[item_counts >= 20].index.tolist()

    if not valid_items:
        logger.warning("No items have enough observations for panel regression")
        return {}

    if len(valid_items) > max_items:
        valid_items = valid_items[:max_items]
        logger.info("Capped panel regression to top %d items", max_items)

    sub = sub[sub[item_col].isin(valid_items)].copy()

    if len(sub) < min_rows:
        logger.warning("Too few rows for panel regression: %d", len(sub))
        return {}

    sub["log_demand"] = np.log1p(sub[target_col].clip(lower=0))
    sub = sub.set_index([item_col, date_col]).sort_index()

    if sub.index.duplicated().any():
        sub = sub[~sub.index.duplicated(keep="first")]

    try:
        fit = PanelOLS(
            dependent=sub["log_demand"],
            exog=sub[[promo_col]],
            entity_effects=True,
            time_effects=False,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        logger.error("Panel regression failed: %s", exc)
        return {}

    coef = float(fit.params[promo_col])
    ci = fit.conf_int().loc[promo_col]
    p_value = float(fit.pvalues[promo_col])

    result = {
        "promotion_coef": round(coef, 4),
        "pct_demand_change": round((np.exp(coef) - 1.0) * 100.0, 2),
        "std_error": round(float(fit.std_errors[promo_col]), 4),
        "p_value": round(p_value, 4),
        "ci_low": round(float(ci["lower"]), 4),
        "ci_high": round(float(ci["upper"]), 4),
        "n_obs": int(len(sub)),
        "n_items": int(len(valid_items)),
        "significant": bool(p_value < 0.05),
        "rsquared_within": round(float(fit.rsquared), 4),
    }

    logger.info(
        "Panel sensitivity | coef=%.4f | pct=%.2f%% | p=%.4f | items=%d",
        result["promotion_coef"],
        result["pct_demand_change"],
        result["p_value"],
        result["n_items"],
    )
    return result


def compute_revenue_proxy_curve(
    baseline_demand: float,
    promotion_coef: float,
    revenue_per_unit: float = 1.0,
) -> pd.DataFrame:
    """
    Build a simple revenue proxy comparison for promotion OFF vs ON.

    Since real prices are not available, revenue is represented as:
        expected_demand * revenue_per_unit
    """
    rows = []

    for state in [0, 1]:
        multiplier = np.exp(promotion_coef) if state == 1 else 1.0
        expected_demand = baseline_demand * multiplier

        rows.append(
            {
                "promotion_on": int(state),
                "expected_demand": round(float(expected_demand), 4),
                "revenue_proxy": round(float(expected_demand * revenue_per_unit), 4),
                "demand_delta_pct": round(float((multiplier - 1.0) * 100.0), 2),
            }
        )

    return pd.DataFrame(rows)