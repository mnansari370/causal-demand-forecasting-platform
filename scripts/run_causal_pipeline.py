from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.causal.causal_forest import prepare_causal_forest_data, run_causal_forest
from src.causal.causal_plots import plot_did_summary, plot_hte_table
from src.causal.did_estimator import (
    naive_vs_did_comparison,
    prepare_did_data,
    run_did,
    run_placebo_test,
)
from src.data.load_data import load_config
from src.features.build_features import get_feature_columns
from src.utils.logger import get_logger


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)

    logger = get_logger(
        "run_causal_pipeline",
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

    logger.info("=" * 60)
    logger.info("STAGE 1: Loading training data")
    logger.info("=" * 60)

    train_path = processed_dir / "train_features.parquet"
    val_path = processed_dir / "val_features.parquet"

    if not train_path.exists():
        logger.error("train_features.parquet not found. Run build_forecasting_features.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_df = full_df.sort_values([store_col, item_col, date_col]).reset_index(drop=True)

    logger.info("Combined train+val: %s rows", len(full_df))
    logger.info("Date range: %s to %s", full_df[date_col].min(), full_df[date_col].max())
    logger.info(
        "Promotion rate: %.3f (%.0f%% of rows are on promotion)",
        full_df[treatment_col].mean(),
        full_df[treatment_col].mean() * 100,
    )

    logger.info("=" * 60)
    logger.info("STAGE 2: Naive estimate")
    logger.info("=" * 60)

    naive_result = naive_vs_did_comparison(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        store_col=store_col,
        item_col=item_col,
        date_col=date_col,
        post_start="2016-09-01",
        post_end="2016-12-31",
    )

    logger.info("=" * 60)
    logger.info("STAGE 3: Preparing DiD panel")
    logger.info("=" * 60)

    did_panel = prepare_did_data(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        pre_start="2016-06-01",
        pre_end="2016-08-31",
        post_start="2016-09-01",
        post_end="2016-12-31",
        min_pre_rows=config["causal"]["did_min_pre_periods"],
        min_post_rows=config["causal"]["did_min_post_periods"],
    )

    logger.info("DiD panel shape: %s", did_panel.shape)

    logger.info("=" * 60)
    logger.info("STAGE 4: Running DiD")
    logger.info("=" * 60)

    did_result = run_did(did_panel, label="DiD Promotion Effect")

    if not did_result:
        logger.error("DiD failed — check panel data")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STAGE 5: Placebo test")
    logger.info("=" * 60)

    placebo_result = run_placebo_test(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        placebo_pre_start="2016-06-01",
        placebo_pre_end="2016-07-15",
        placebo_post_start="2016-07-16",
        placebo_post_end="2016-08-31",
        real_did_estimate=did_result["estimate"],
    )

    placebo_verdict = "PASSED" if placebo_result.get("passed") else "FAILED"

    logger.info("=" * 60)
    logger.info("STAGE 6: Causal Forest")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(full_df, config)
    feature_cols = [c for c in feature_cols if c != treatment_col]

    Y, T, X, meta_df = prepare_causal_forest_data(
        df=full_df,
        treatment_col=treatment_col,
        target_col=target_col,
        date_col=date_col,
        store_col=store_col,
        item_col=item_col,
        feature_cols=feature_cols,
        start_date="2016-09-01",
        end_date="2016-12-31",
    )

    store_hte, item_hte = run_causal_forest(
        Y=Y,
        T=T,
        X=X,
        meta_df=meta_df,
        store_col=store_col,
        item_col=item_col,
        n_estimators=config["causal"]["causal_forest_n_estimators"],
        random_state=config["project"]["random_seed"],
    )

    logger.info("=" * 60)
    logger.info("STAGE 7: Saving results and figures")
    logger.info("=" * 60)

    def _safe_json(d: dict) -> dict:
        return {
            k: (None if v is None or (isinstance(v, float) and not (v == v)) else v)
            for k, v in d.items()
        }

    (results_dir / "causal_did_result.json").write_text(
        json.dumps(_safe_json(did_result), indent=2)
    )
    (results_dir / "causal_placebo_result.json").write_text(
        json.dumps(_safe_json(placebo_result), indent=2)
    )
    (results_dir / "causal_naive_comparison.json").write_text(
        json.dumps(_safe_json(naive_result), indent=2)
    )

    store_hte.to_csv(results_dir / "causal_store_hte.csv", index=False)
    item_hte.to_csv(results_dir / "causal_item_hte.csv", index=False)

    plot_did_summary(
        did_result=did_result,
        placebo_result=placebo_result,
        naive_result=naive_result,
        save_path=figures_dir / "causal_did_summary.png",
    )

    plot_hte_table(
        hte_df=store_hte,
        id_col=store_col,
        title="Store-Level Promotion Sensitivity (Causal Forest HTE)",
        save_path=figures_dir / "causal_store_hte.png",
    )

    plot_hte_table(
        hte_df=item_hte,
        id_col=item_col,
        title="Item-Level Promotion Sensitivity (Top 15 vs Bottom 15)",
        save_path=figures_dir / "causal_item_hte.png",
    )

    print("\n" + "=" * 70)
    print("WEEK 3 CAUSAL INFERENCE RESULTS")
    print("=" * 70)

    print(f"\nNaive estimate (biased):    {naive_result['naive_estimate']:+.4f} units/day")
    print(f"DiD estimate (causal ATT):  {did_result['estimate']:+.4f} units/day")
    print(f"95% CI:                     [{did_result['ci_low']:+.4f}, {did_result['ci_high']:+.4f}]")
    print(f"p-value:                    {did_result['p_value']:.4f}")
    print(f"Significant (p<0.05):       {did_result['significant']}")

    print(f"\nPlacebo test:               {placebo_verdict}")
    if placebo_result.get("estimate") is not None:
        print(f"Placebo estimate:           {placebo_result['estimate']:+.4f} (should be ~0)")

    print(f"\nCausal Forest HTE:")
    print(f"  Stores analysed:          {len(store_hte)}")
    print(f"  Items analysed:           {len(item_hte)}")

    print("\n" + "=" * 70)
    print("All outputs saved to outputs/evaluation/ and outputs/figures/")
    print("=" * 70 + "\n")

    logger.info("Week 3 causal pipeline complete")


if __name__ == "__main__":
    main()