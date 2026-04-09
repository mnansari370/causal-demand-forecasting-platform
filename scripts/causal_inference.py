"""
Causal inference pipeline for estimating the true promotion effect.

This script runs:
- a naive promoted vs non-promoted comparison,
- Difference-in-Differences,
- a placebo robustness test,
- a causal forest for heterogeneous treatment effects.

The purpose is to separate true incremental lift from simple correlation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.causal.causal_forest import prepare_causal_forest_data, run_causal_forest
from src.causal.causal_plots import plot_did_summary, plot_hte_ranking
from src.causal.did_estimator import (
    naive_vs_did_comparison,
    prepare_did_data,
    run_did,
    run_placebo_test,
)
from src.data.load_data import load_config
from src.features.build_features import get_feature_columns
from src.utils.logger import get_logger


def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "causal_inference",
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
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    treatment_col = config["causal"]["treatment_column"]

    for split in ["train_features", "val_features"]:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            logger.error("%s not found. Run scripts/feature_engineering.py first.", path)
            sys.exit(1)

    full_df = (
        pd.concat(
            [
                pd.read_parquet(processed_dir / "train_features.parquet"),
                pd.read_parquet(processed_dir / "val_features.parquet"),
            ],
            ignore_index=True,
        )
        .sort_values([store_col, item_col, date_col])
        .reset_index(drop=True)
    )

    logger.info(
        "Combined rows: %d | date range: %s to %s",
        len(full_df),
        full_df[date_col].min().date(),
        full_df[date_col].max().date(),
    )
    logger.info("Promotion rate: %.1f%%", full_df[treatment_col].mean() * 100)

    naive_result = naive_vs_did_comparison(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        store_col=store_col,
        item_col=item_col,
        date_col=date_col,
        post_start="2016-10-01",
        post_end="2016-12-31",
    )
    if not naive_result:
        logger.error("Naive comparison failed")
        sys.exit(1)

    did_panel = prepare_did_data(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        pre_start="2016-07-01",
        pre_end="2016-09-30",
        post_start="2016-10-01",
        post_end="2016-12-31",
        min_pre_rows=config["causal"]["did_min_pre_periods"],
        min_post_rows=config["causal"]["did_min_post_periods"],
    )

    if len(did_panel) < 10:
        logger.error("DiD panel too small: %d rows", len(did_panel))
        sys.exit(1)

    did_result = run_did(did_panel, label="DiD — Q3 to Q4 2016")
    if not did_result:
        logger.error("DiD regression failed")
        sys.exit(1)

    placebo_result = run_placebo_test(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        placebo_pre_start="2016-07-01",
        placebo_pre_end="2016-07-31",
        placebo_post_start="2016-08-01",
        placebo_post_end="2016-09-30",
        real_did_estimate=did_result["estimate"],
        min_pre_rows=config["causal"]["did_min_pre_periods"],
        min_post_rows=config["causal"]["did_min_post_periods"],
    )

    feature_cols = [c for c in get_feature_columns(full_df, config) if c != treatment_col]

    Y, T, X, meta_df = prepare_causal_forest_data(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        feature_cols=feature_cols,
        start_date="2016-10-01",
        end_date="2016-12-31",
    )

    store_hte = pd.DataFrame()
    item_hte = pd.DataFrame()

    if len(Y) >= 50:
        store_hte, item_hte = run_causal_forest(
            Y,
            T,
            X,
            meta_df,
            store_col=store_col,
            item_col=item_col,
            n_estimators=config["causal"]["causal_forest_n_estimators"],
            random_state=config["project"]["random_seed"],
        )
    else:
        logger.warning("Too few rows for causal forest (%d)", len(Y))

    (results_dir / "causal_did_result.json").write_text(json.dumps(_safe_json(did_result), indent=2))
    (results_dir / "causal_placebo_result.json").write_text(json.dumps(_safe_json(placebo_result), indent=2))
    (results_dir / "causal_naive_comparison.json").write_text(json.dumps(_safe_json(naive_result), indent=2))

    if not store_hte.empty:
        store_hte.to_csv(results_dir / "causal_store_hte.csv", index=False)
    if not item_hte.empty:
        item_hte.to_csv(results_dir / "causal_item_hte.csv", index=False)

    plot_did_summary(
        did_result,
        placebo_result,
        naive_result,
        save_path=figures_dir / "causal_did_summary.png",
    )

    if not store_hte.empty:
        plot_hte_ranking(
            store_hte,
            id_col=store_col,
            top_n=10,
            title="Store Promotion Sensitivity (Causal Forest HTE)",
            save_path=figures_dir / "causal_store_hte.png",
        )

    if not item_hte.empty:
        plot_hte_ranking(
            item_hte,
            id_col=item_col,
            top_n=10,
            title="Item Promotion Sensitivity (Top 10 vs Bottom 10)",
            save_path=figures_dir / "causal_item_hte.png",
        )

    print("\n" + "=" * 70)
    print("CAUSAL INFERENCE RESULTS")
    print("=" * 70)
    print(f"Naive estimate (biased):   {naive_result['naive_estimate']:+.4f} units/day")
    print(f"DiD ATT (causal):          {did_result['estimate']:+.4f} units/day")
    print(f"95% CI:                    [{did_result['ci_low']:+.4f}, {did_result['ci_high']:+.4f}]")
    print(f"p-value:                   {did_result['p_value']:.4f}")
    print(f"Significant:               {did_result['significant']}")

    if naive_result["naive_estimate"] != 0:
        bias = abs(
            (naive_result["naive_estimate"] - did_result["estimate"])
            / naive_result["naive_estimate"]
            * 100
        )
        print(f"Selection bias:            {bias:.1f}% overestimation in naive analysis")

    print(f"\nPlacebo verdict:           {placebo_result.get('verdict', 'N/A')}")

    if not store_hte.empty:
        print(f"Stores with HTE:           {len(store_hte)}")
    if not item_hte.empty:
        print(f"Items with HTE:            {len(item_hte)}")
    print("=" * 70 + "\n")

    logger.info("Causal inference pipeline complete")


if __name__ == "__main__":
    main()