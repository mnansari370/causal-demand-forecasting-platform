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
    Scenario using ONLY panel promotion coefficient.

    If promotion is ON:
        demand = baseline * exp(promotion_coef)

    If promotion is OFF:
        demand = baseline
    """
    if run_promotion:
        multiplier = float(np.exp(promotion_coef))
        demand_q05 = baseline_demand_q05 * multiplier
        demand_q50 = baseline_demand_q50 * multiplier
        demand_q95 = baseline_demand_q95 * multiplier
    else:
        demand_q05 = baseline_demand_q05
        demand_q50 = baseline_demand_q50
        demand_q95 = baseline_demand_q95

    demand_q05 = max(0.0, demand_q05)
    demand_q50 = max(0.0, demand_q50)
    demand_q95 = max(0.0, demand_q95)

    revenue_q05 = demand_q05 * revenue_per_unit * horizon_days
    revenue_q50 = demand_q50 * revenue_per_unit * horizon_days
    revenue_q95 = demand_q95 * revenue_per_unit * horizon_days

    baseline_revenue = baseline_demand_q50 * revenue_per_unit * horizon_days
    revenue_delta = revenue_q50 - baseline_revenue
    revenue_delta_pct = (revenue_q50 / baseline_revenue - 1.0) * 100.0 if baseline_revenue > 0 else 0.0

    return {
        "method": "panel",
        "run_promotion": run_promotion,
        "promotion_coef_used": round(promotion_coef, 4),
        "baseline_demand_q50": round(baseline_demand_q50, 4),
        "expected_demand_q05": round(demand_q05, 4),
        "expected_demand_q50": round(demand_q50, 4),
        "expected_demand_q95": round(demand_q95, 4),
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
    Scenario using ONLY DiD additive promotion lift.

    If promotion is ON:
        demand = baseline + promotion_lift

    If promotion is OFF:
        demand = baseline
    """
    if run_promotion:
        demand_q05 = baseline_demand_q05 + promotion_lift
        demand_q50 = baseline_demand_q50 + promotion_lift
        demand_q95 = baseline_demand_q95 + promotion_lift
    else:
        demand_q05 = baseline_demand_q05
        demand_q50 = baseline_demand_q50
        demand_q95 = baseline_demand_q95

    demand_q05 = max(0.0, demand_q05)
    demand_q50 = max(0.0, demand_q50)
    demand_q95 = max(0.0, demand_q95)

    revenue_q05 = demand_q05 * revenue_per_unit * horizon_days
    revenue_q50 = demand_q50 * revenue_per_unit * horizon_days
    revenue_q95 = demand_q95 * revenue_per_unit * horizon_days

    baseline_revenue = baseline_demand_q50 * revenue_per_unit * horizon_days
    revenue_delta = revenue_q50 - baseline_revenue
    revenue_delta_pct = (revenue_q50 / baseline_revenue - 1.0) * 100.0 if baseline_revenue > 0 else 0.0

    return {
        "method": "did",
        "run_promotion": run_promotion,
        "promotion_lift_used": round(promotion_lift, 4),
        "baseline_demand_q50": round(baseline_demand_q50, 4),
        "expected_demand_q05": round(demand_q05, 4),
        "expected_demand_q50": round(demand_q50, 4),
        "expected_demand_q95": round(demand_q95, 4),
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
    Run four scenarios:
      - panel, no promotion
      - panel, promotion
      - did, no promotion
      - did, promotion
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
    logger.info("Scenario comparison complete: %d scenarios", len(df))
    return df
