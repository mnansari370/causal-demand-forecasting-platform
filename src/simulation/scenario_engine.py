"""
Scenario simulation engine.

This module combines forecast baselines with promotion-effect estimates to
answer simple what-if questions such as:

- What happens if a promotion is run?
- What is the expected revenue range over the forecast horizon?
- How does a panel-regression effect compare against a DiD effect?

We keep the panel and DiD scenarios separate to avoid double-counting the
promotion effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def simulate_panel_scenario(
    baseline_demand_q05: float,
    baseline_demand_q50: float,
    baseline_demand_q95: float,
    promotion_coef: float,
    run_promotion: bool,
    revenue_per_unit: float = 1.0,
    horizon_days: int = 14,
) -> dict:
    """
    Scenario using the multiplicative promotion effect from panel regression.

    If promotion is ON:
        demand = baseline * exp(promotion_coef)

    If promotion is OFF:
        demand = baseline
    """
    if run_promotion:
        multiplier = float(np.exp(promotion_coef))
        d05 = max(0.0, baseline_demand_q05 * multiplier)
        d50 = max(0.0, baseline_demand_q50 * multiplier)
        d95 = max(0.0, baseline_demand_q95 * multiplier)
    else:
        d05 = baseline_demand_q05
        d50 = baseline_demand_q50
        d95 = baseline_demand_q95

    baseline_revenue = baseline_demand_q50 * revenue_per_unit * horizon_days
    revenue_q05 = d05 * revenue_per_unit * horizon_days
    revenue_q50 = d50 * revenue_per_unit * horizon_days
    revenue_q95 = d95 * revenue_per_unit * horizon_days

    revenue_delta = revenue_q50 - baseline_revenue
    revenue_delta_pct = (
        (revenue_q50 / baseline_revenue - 1.0) * 100.0
        if baseline_revenue > 0
        else 0.0
    )

    return {
        "method": "panel",
        "run_promotion": run_promotion,
        "promotion_coef_used": round(promotion_coef, 4),
        "baseline_demand_q50": round(baseline_demand_q50, 4),
        "expected_demand_q05": round(d05, 4),
        "expected_demand_q50": round(d50, 4),
        "expected_demand_q95": round(d95, 4),
        "baseline_revenue": round(baseline_revenue, 4),
        "expected_revenue_q05": round(revenue_q05, 4),
        "expected_revenue_q50": round(revenue_q50, 4),
        "expected_revenue_q95": round(revenue_q95, 4),
        "revenue_delta": round(revenue_delta, 4),
        "revenue_delta_pct": round(revenue_delta_pct, 2),
    }


def simulate_did_scenario(
    baseline_demand_q05: float,
    baseline_demand_q50: float,
    baseline_demand_q95: float,
    promotion_lift: float,
    run_promotion: bool,
    revenue_per_unit: float = 1.0,
    horizon_days: int = 14,
) -> dict:
    """
    Scenario using the additive promotion effect from Difference-in-Differences.

    If promotion is ON:
        demand = baseline + promotion_lift

    If promotion is OFF:
        demand = baseline
    """
    if run_promotion:
        d05 = max(0.0, baseline_demand_q05 + promotion_lift)
        d50 = max(0.0, baseline_demand_q50 + promotion_lift)
        d95 = max(0.0, baseline_demand_q95 + promotion_lift)
    else:
        d05 = baseline_demand_q05
        d50 = baseline_demand_q50
        d95 = baseline_demand_q95

    baseline_revenue = baseline_demand_q50 * revenue_per_unit * horizon_days
    revenue_q05 = d05 * revenue_per_unit * horizon_days
    revenue_q50 = d50 * revenue_per_unit * horizon_days
    revenue_q95 = d95 * revenue_per_unit * horizon_days

    revenue_delta = revenue_q50 - baseline_revenue
    revenue_delta_pct = (
        (revenue_q50 / baseline_revenue - 1.0) * 100.0
        if baseline_revenue > 0
        else 0.0
    )

    return {
        "method": "did",
        "run_promotion": run_promotion,
        "promotion_lift_used": round(promotion_lift, 4),
        "baseline_demand_q50": round(baseline_demand_q50, 4),
        "expected_demand_q05": round(d05, 4),
        "expected_demand_q50": round(d50, 4),
        "expected_demand_q95": round(d95, 4),
        "baseline_revenue": round(baseline_revenue, 4),
        "expected_revenue_q05": round(revenue_q05, 4),
        "expected_revenue_q50": round(revenue_q50, 4),
        "expected_revenue_q95": round(revenue_q95, 4),
        "revenue_delta": round(revenue_delta, 4),
        "revenue_delta_pct": round(revenue_delta_pct, 2),
    }


def run_scenario_comparison(
    baseline_demand_q05: float,
    baseline_demand_q50: float,
    baseline_demand_q95: float,
    promotion_coef: float,
    promotion_lift: float,
    revenue_per_unit: float = 1.0,
    horizon_days: int = 14,
) -> pd.DataFrame:
    """
    Build a comparison table for panel-based and DiD-based scenarios
    with and without promotion.
    """
    rows = []

    for run_promotion in [False, True]:
        rows.append(
            simulate_panel_scenario(
                baseline_demand_q05=baseline_demand_q05,
                baseline_demand_q50=baseline_demand_q50,
                baseline_demand_q95=baseline_demand_q95,
                promotion_coef=promotion_coef,
                run_promotion=run_promotion,
                revenue_per_unit=revenue_per_unit,
                horizon_days=horizon_days,
            )
        )

        rows.append(
            simulate_did_scenario(
                baseline_demand_q05=baseline_demand_q05,
                baseline_demand_q50=baseline_demand_q50,
                baseline_demand_q95=baseline_demand_q95,
                promotion_lift=promotion_lift,
                run_promotion=run_promotion,
                revenue_per_unit=revenue_per_unit,
                horizon_days=horizon_days,
            )
        )

    df = pd.DataFrame(rows)
    logger.info("Scenario comparison complete: %d rows", len(df))
    return df