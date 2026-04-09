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
    Estimate promotion sensitivity per product family.

    Model:
        log1p(unit_sales) ~ onpromotion + controls

    Interpretation:
        onpromotion coefficient is the approximate demand uplift when promotion is active.
        pct_demand_change = exp(beta) - 1

    This is NOT true price elasticity because no real price column exists.
    """

    if family_col not in df.columns:
        logger.warning("Column '%s' not found. Cannot estimate family-level sensitivity.", family_col)
        return pd.DataFrame()

    available_controls = [c for c in control_cols if c in df.columns]
    missing_controls = [c for c in control_cols if c not in df.columns]

    if missing_controls:
        logger.info("Missing controls skipped: %s", missing_controls)

    families = sorted(df[family_col].dropna().unique().tolist())
    results = []

    logger.info("Estimating promotion sensitivity for %d families", len(families))

    for family in families:
        sub = df[df[family_col] == family].copy()
        sub = sub.dropna(subset=[target_col, promo_col])

        if len(sub) < min_obs:
            logger.info("Skipping family '%s' (n=%d < %d)", family, len(sub), min_obs)
            continue

        sub["log_demand"] = np.log1p(sub[target_col].clip(lower=0))

        rhs_terms = [promo_col] + available_controls
        formula = "log_demand ~ " + " + ".join(rhs_terms)

        try:
            model = smf.ols(formula, data=sub)
            fit = model.fit()
        except Exception as exc:
            logger.warning("Family regression failed for '%s': %s", family, exc)
            continue

        if promo_col not in fit.params:
            logger.warning("Promotion coefficient missing for family '%s'", family)
            continue

        coef = float(fit.params[promo_col])
        se = float(fit.bse[promo_col])
        p_value = float(fit.pvalues[promo_col])
        ci = fit.conf_int().loc[promo_col]
        pct_effect = (np.exp(coef) - 1.0) * 100.0

        results.append(
            {
                "family": family,
                "promotion_coef": round(coef, 4),
                "pct_demand_change": round(pct_effect, 2),
                "std_error": round(se, 4),
                "p_value": round(p_value, 4),
                "ci_low": round(float(ci.iloc[0]), 4),
                "ci_high": round(float(ci.iloc[1]), 4),
                "n_obs": int(len(sub)),
                "significant": bool(p_value < 0.05),
                "r_squared": round(float(fit.rsquared), 4),
            }
        )

    if not results:
        logger.warning("No family-level promotion sensitivity estimates produced")
        return pd.DataFrame()

    result_df = (
        pd.DataFrame(results)
        .sort_values("promotion_coef", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Family-level sensitivity complete | families=%d | significant=%d",
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
    Estimate promotion sensitivity using panel regression with item fixed effects.

    Model:
        log1p(unit_sales_it) ~ onpromotion_it + item fixed effects

    This is a stronger estimate than naive regression because it compares
    promoted vs non-promoted periods within the same item.
    """

    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        logger.warning("linearmodels not installed. Skipping panel regression.")
        return {}

    sub = df.dropna(subset=[target_col, promo_col, item_col, date_col]).copy()
    sub = sub[sub[target_col] >= 0].copy()

    item_counts = sub.groupby(item_col)[target_col].count().sort_values(ascending=False)
    valid_items = item_counts[item_counts >= 20].index.tolist()

    if len(valid_items) == 0:
        logger.warning("No valid items available for panel regression")
        return {}

    if len(valid_items) > max_items:
        valid_items = valid_items[:max_items]
        logger.info("Capped panel regression to top %d items by observation count", max_items)

    sub = sub[sub[item_col].isin(valid_items)].copy()

    if len(sub) < min_rows:
        logger.warning("Too few rows for panel regression: %d", len(sub))
        return {}

    sub["log_demand"] = np.log1p(sub[target_col].clip(lower=0))
    sub = sub.set_index([item_col, date_col]).sort_index()

    if sub.index.duplicated().any():
        logger.info("Dropping duplicated (item, date) rows for panel regression")
        sub = sub[~sub.index.duplicated(keep="first")]

    try:
        model = PanelOLS(
            dependent=sub["log_demand"],
            exog=sub[[promo_col]],
            entity_effects=True,
            time_effects=False,
        )
        fit = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        logger.error("Panel promotion sensitivity failed: %s", exc)
        return {}

    coef = float(fit.params[promo_col])
    se = float(fit.std_errors[promo_col])
    p_value = float(fit.pvalues[promo_col])
    ci = fit.conf_int().loc[promo_col]
    pct_effect = (np.exp(coef) - 1.0) * 100.0

    result = {
        "promotion_coef": round(coef, 4),
        "pct_demand_change": round(pct_effect, 2),
        "std_error": round(se, 4),
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
    promotion_states: list[int] | None = None,
    revenue_per_unit: float = 1.0,
) -> pd.DataFrame:
    """
    Build a simple revenue proxy comparison for promotion OFF vs ON.

    Since we do not have real prices, this returns a revenue proxy:
        expected_demand * revenue_per_unit

    Promotion coefficient is from:
        log1p(unit_sales) ~ onpromotion + controls
    """

    if promotion_states is None:
        promotion_states = [0, 1]

    rows = []

    for state in promotion_states:
        if state == 1:
            demand_multiplier = np.exp(promotion_coef)
        else:
            demand_multiplier = 1.0

        expected_demand = baseline_demand * demand_multiplier
        revenue_proxy = expected_demand * revenue_per_unit

        rows.append(
            {
                "promotion_on": int(state),
                "expected_demand": round(float(expected_demand), 4),
                "revenue_proxy": round(float(revenue_proxy), 4),
                "demand_delta_pct": round(float((demand_multiplier - 1.0) * 100.0), 2),
            }
        )
    return pd.DataFrame(rows)
