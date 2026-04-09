"""
Promotion sensitivity and scenario analysis pipeline.

This script estimates:
- family-level promotion sensitivity,
- panel fixed-effects promotion sensitivity,
- simple revenue proxy comparisons,
- promotion scenarios using both panel and DiD estimates.

We call it promotion sensitivity rather than price elasticity because the
dataset does not include actual prices.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.promotion_analysis.promotion_plots import (
    plot_family_promotion_sensitivity,
    plot_revenue_proxy_curve,
    plot_scenario_comparison,
    plot_simulation_output,
)
from src.promotion_analysis.promotion_sensitivity import (
    compute_revenue_proxy_curve,
    estimate_family_promotion_sensitivity,
    estimate_panel_promotion_sensitivity,
)
from src.simulation.scenario_engine import (
    run_scenario_comparison,
    simulate_did_scenario,
    simulate_panel_scenario,
)
from src.utils.logger import get_logger


def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "promotion_analysis",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_col = config["data"]["target_column"]
    date_col = config["data"]["date_column"]
    item_col = config["data"]["item_column"]
    promo_col = config["data"]["promo_column"]

    full_df = pd.concat(
        [
            pd.read_parquet(processed_dir / "train_features.parquet"),
            pd.read_parquet(processed_dir / "val_features.parquet"),
        ],
        ignore_index=True,
    )

    family_col = "family" if "family" in full_df.columns else item_col
    logger.info("Combined shape: %s | family_col=%s", full_df.shape, family_col)

    control_cols = [
        "day_of_week",
        "month",
        "is_holiday",
        "oil_price",
        "unit_sales_rolling_mean_7d",
        "year",
    ]

    sensitivity_df = estimate_family_promotion_sensitivity(
        df=full_df,
        target_col=target_col,
        promo_col=promo_col,
        family_col=family_col,
        control_cols=control_cols,
        min_obs=100,
    )

    if not sensitivity_df.empty:
        sensitivity_df.to_csv(results_dir / "promotion_sensitivity_by_family.csv", index=False)
        plot_family_promotion_sensitivity(
            sensitivity_df=sensitivity_df,
            top_n=min(20, len(sensitivity_df)),
            save_path=figures_dir / "promotion_sensitivity_by_family.png",
        )

    panel_result = estimate_panel_promotion_sensitivity(
        df=full_df,
        target_col=target_col,
        promo_col=promo_col,
        date_col=date_col,
        item_col=item_col,
        min_rows=500,
        max_items=200,
    )

    (results_dir / "panel_promotion_sensitivity.json").write_text(
        json.dumps(_safe_json(panel_result), indent=2)
    )

    baseline_demand = float(full_df[target_col].mean())
    selected_coef = 0.0
    selected_label = "All items"

    if not sensitivity_df.empty:
        sig_df = sensitivity_df[sensitivity_df["significant"]]
        top_row = sig_df.iloc[0] if not sig_df.empty else sensitivity_df.iloc[0]

        selected_label = str(top_row["family"])
        selected_coef = float(top_row["promotion_coef"])

        group_df = full_df[full_df[family_col] == selected_label]
        if len(group_df) > 0:
            baseline_demand = float(group_df[target_col].mean())

        revenue_df = compute_revenue_proxy_curve(
            baseline_demand=baseline_demand,
            promotion_coef=selected_coef,
        )
        revenue_df.to_csv(results_dir / "promotion_revenue_proxy_curve.csv", index=False)

        plot_revenue_proxy_curve(
            revenue_df,
            label_name=selected_label,
            save_path=figures_dir / "promotion_revenue_proxy_curve.png",
        )

    q_path = results_dir / "lgbm_quantile_predictions.parquet"
    did_path = results_dir / "causal_did_result.json"

    if q_path.exists():
        q_df = pd.read_parquet(q_path)
        q05 = float(q_df["q0.05"].mean())
        q50 = float(q_df["q0.5"].mean())
        q95 = float(q_df["q0.95"].mean())
    else:
        logger.warning("Quantile predictions not found — using fallback demand quantiles")
        q05 = float(full_df[target_col].quantile(0.05))
        q50 = float(full_df[target_col].mean())
        q95 = float(full_df[target_col].quantile(0.95))

    promotion_lift = 0.0
    if did_path.exists():
        promotion_lift = float(json.loads(did_path.read_text()).get("estimate", 0.0) or 0.0)

    panel_coef = float(panel_result.get("promotion_coef", selected_coef)) if panel_result else selected_coef

    logger.info(
        "Scenario inputs | panel_coef=%.4f | did_lift=%.4f | q50=%.4f",
        panel_coef,
        promotion_lift,
        q50,
    )

    scenario_df = run_scenario_comparison(q05, q50, q95, panel_coef, promotion_lift)
    scenario_df.to_csv(results_dir / "scenario_grid.csv", index=False)

    best_panel = simulate_panel_scenario(q05, q50, q95, panel_coef, run_promotion=True)
    best_did = simulate_did_scenario(q05, q50, q95, promotion_lift, run_promotion=True)

    plot_scenario_comparison(
        scenario_df[scenario_df["run_promotion"] == True].copy(),
        save_path=figures_dir / "scenario_comparison.png",
    )
    plot_simulation_output(best_panel, save_path=figures_dir / "simulation_panel.png")
    plot_simulation_output(best_did, save_path=figures_dir / "simulation_did.png")

    (results_dir / "best_scenario_panel.json").write_text(json.dumps(_safe_json(best_panel), indent=2))
    (results_dir / "best_scenario_did.json").write_text(json.dumps(_safe_json(best_did), indent=2))

    print("\n" + "=" * 70)
    print("PROMOTION SENSITIVITY & SCENARIO RESULTS")
    print("=" * 70)
    if not sensitivity_df.empty:
        print(f"Families estimated: {len(sensitivity_df)} | Significant: {int(sensitivity_df['significant'].sum())}")
        print("\nTop 5 families by promotion sensitivity:")
        print(sensitivity_df[["family", "pct_demand_change", "p_value"]].head(5).to_string(index=False))

    if panel_result:
        print(f"\nPanel OLS coefficient:  {panel_result['promotion_coef']:+.4f}")
        print(f"Demand uplift:          {panel_result['pct_demand_change']:+.2f}%")
        print(f"p-value:                {panel_result['p_value']:.4f}")

    print(f"\nPanel scenario revenue delta (14-day):  {best_panel['revenue_delta_pct']:+.2f}%")
    print(f"DiD scenario revenue delta (14-day):    {best_did['revenue_delta_pct']:+.2f}%")
    print("=" * 70 + "\n")

    logger.info("Promotion analysis pipeline complete")


if __name__ == "__main__":
    main()