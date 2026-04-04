from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.evaluation.evaluate import (
    build_results_table,
    evaluate_point_forecast,
    evaluate_probabilistic_forecast,
    plot_calibration,
    plot_feature_importance,
    plot_forecast_with_intervals,
    save_results,
)
from src.features.build_features import (
    add_promotion_features,
    add_target_encoding,
    get_feature_columns,
)
from src.forecasting.lgbm_forecaster import LGBMPointForecaster
from src.forecasting.lgbm_quantile import LGBMQuantileForecaster
from src.forecasting.xgb_forecaster import XGBPointForecaster
from src.utils.logger import get_logger


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    logger = get_logger(
        "run_forecasting_pipeline",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]
    models_dir = PROJECT_ROOT / config["outputs"]["models_dir"]

    for d in [results_dir, figures_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    target_col = config["data"]["target_column"]
    date_col = config["data"]["date_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    promo_col = config["data"]["promo_column"]

    logger.info("=" * 60)
    logger.info("STAGE 1: Loading feature parquets")
    logger.info("=" * 60)

    for split in ["train_features", "val_features", "test_features"]:
        p = processed_dir / f"{split}.parquet"
        if not p.exists():
            logger.error("%s not found. Run scripts/build_forecasting_features.py first.", p)
            sys.exit(1)

    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    val_df = pd.read_parquet(processed_dir / "val_features.parquet")
    test_df = pd.read_parquet(processed_dir / "test_features.parquet")

    logger.info("Loaded shapes | Train=%s | Val=%s | Test=%s", train_df.shape, val_df.shape, test_df.shape)

    logger.info("=" * 60)
    logger.info("STAGE 2: Promotion features + target encoding")
    logger.info("=" * 60)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values([store_col, item_col, date_col]).reset_index(drop=True)

    if promo_col in full_df.columns:
        full_df = add_promotion_features(
            full_df,
            promo_col=promo_col,
            group_cols=[store_col, item_col],
            date_col=date_col,
        )
    else:
        logger.warning("Promo column '%s' not found. Skipping promotion features.", promo_col)

    train_mask = full_df[date_col] <= pd.to_datetime(config["splits"]["train_end"])

    if config["features"].get("use_target_encoding"):
        full_df = add_target_encoding(
            full_df,
            target_col=target_col,
            categorical_cols=config["features"].get("target_encoding_cols", []),
            train_mask=train_mask,
        )

    splits = config["splits"]
    train_df = full_df[full_df[date_col] <= pd.to_datetime(splits["train_end"])].copy()
    val_df = full_df[
        (full_df[date_col] >= pd.to_datetime(splits["val_start"])) &
        (full_df[date_col] <= pd.to_datetime(splits["val_end"]))
    ].copy()
    test_df = full_df[
        (full_df[date_col] >= pd.to_datetime(splits["test_start"])) &
        (full_df[date_col] <= pd.to_datetime(splits["test_end"]))
    ].copy()

    logger.info("After new features | Train=%s | Val=%s | Test=%s", train_df.shape, val_df.shape, test_df.shape)

    logger.info("=" * 60)
    logger.info("STAGE 3: Building X/y matrices")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(train_df, config)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col].values

    logger.info("X_train=%s | X_val=%s | X_test=%s", X_train.shape, X_val.shape, X_test.shape)
    logger.info("Using %d features", len(feature_cols))

    all_results: list[dict] = []

    logger.info("=" * 60)
    logger.info("STAGE 4: Loading Week 1 baseline")
    logger.info("=" * 60)

    baseline_path = results_dir / "week1_baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            week1_results = json.load(f)

        for row in week1_results:
            if row.get("evaluation_split") == "test":
                all_results.append(
                    {
                        "model": row["model"],
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "mape": row["mape"],
                        "coverage_90": None,
                        "interval_width": None,
                        "n_samples": len(y_test),
                    }
                )
                logger.info(
                    "Loaded baseline test result | RMSE=%.4f | MAE=%.4f | MAPE=%.2f%%",
                    row["rmse"],
                    row["mae"],
                    row["mape"],
                )
    else:
        logger.warning("Week 1 baseline not found at %s", baseline_path)

    logger.info("=" * 60)
    logger.info("STAGE 5: LightGBM Point Forecast")
    logger.info("=" * 60)

    lgbm_point = LGBMPointForecaster()
    lgbm_point.fit(X_train, y_train, X_val, y_val)
    lgbm_point.save(models_dir / "lgbm_point.pkl")

    lgbm_preds = lgbm_point.predict(X_test)
    all_results.append(evaluate_point_forecast(y_test, lgbm_preds, "LightGBM Point"))

    fi_lgbm = lgbm_point.feature_importance()
    fi_lgbm.to_csv(results_dir / "lgbm_feature_importance.csv", index=False)
    plot_feature_importance(
        fi_lgbm,
        title="LightGBM Feature Importance",
        save_path=figures_dir / "lgbm_feature_importance.png",
    )

    logger.info("=" * 60)
    logger.info("STAGE 6: LightGBM Quantile Forecast")
    logger.info("=" * 60)

    lgbm_q = LGBMQuantileForecaster(quantiles=[0.1, 0.5, 0.9])
    lgbm_q.fit(X_train, y_train, X_val, y_val)
    lgbm_q.save(models_dir / "lgbm_quantile")

    q_preds = lgbm_q.predict(X_test)

    all_results.append(
        evaluate_probabilistic_forecast(
            y_test,
            q_preds["q0.1"],
            q_preds["q0.5"],
            q_preds["q0.9"],
            "LightGBM Quantile (Q0.1/0.5/0.9)",
        )
    )

    q_preds_df = pd.DataFrame(
        {
            date_col: test_df[date_col].values,
            store_col: test_df[store_col].values,
            item_col: test_df[item_col].values,
            "actual": y_test,
            "q0.1": q_preds["q0.1"],
            "q0.5": q_preds["q0.5"],
            "q0.9": q_preds["q0.9"],
        }
    )
    q_preds_df.to_parquet(results_dir / "lgbm_quantile_predictions.parquet", index=False)

    plot_calibration(
        y_test,
        q_preds,
        title="LightGBM Quantile Calibration",
        save_path=figures_dir / "lgbm_quantile_calibration.png",
    )

    first_store = test_df[store_col].iloc[0]
    first_item = test_df[item_col].iloc[0]
    sample_mask = (test_df[store_col] == first_store) & (test_df[item_col] == first_item)

    if sample_mask.sum() > 5:
        plot_forecast_with_intervals(
            dates=test_df.loc[sample_mask, date_col],
            y_true=test_df.loc[sample_mask, target_col].values,
            y_pred_q10=q_preds["q0.1"][sample_mask.values],
            y_pred_q50=q_preds["q0.5"][sample_mask.values],
            y_pred_q90=q_preds["q0.9"][sample_mask.values],
            title=f"LightGBM Quantile — Store {first_store} Item {first_item}",
            save_path=figures_dir / "sample_quantile_forecast.png",
        )

    logger.info("=" * 60)
    logger.info("STAGE 7: XGBoost Point Forecast")
    logger.info("=" * 60)

    xgb_model = XGBPointForecaster()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_model.save(models_dir / "xgb_point.pkl")

    xgb_preds = xgb_model.predict(X_test)
    all_results.append(evaluate_point_forecast(y_test, xgb_preds, "XGBoost Point"))

    fi_xgb = xgb_model.feature_importance()
    fi_xgb.to_csv(results_dir / "xgb_feature_importance.csv", index=False)
    plot_feature_importance(
        fi_xgb,
        title="XGBoost Feature Importance",
        save_path=figures_dir / "xgb_feature_importance.png",
    )

    logger.info("=" * 60)
    logger.info("STAGE 8: Saving results table")
    logger.info("=" * 60)

    results_df = build_results_table(all_results)
    save_results(results_df, results_dir, name="week2_forecasting_results")

    logger.info("=" * 60)
    logger.info("Week 2 core forecasting pipeline complete")
    logger.info("Results: %s", results_dir)
    logger.info("Models : %s", models_dir)
    logger.info("Figures: %s", figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()