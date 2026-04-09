"""
Train and evaluate the forecasting models.

Models included:
- Seasonal Naive baseline (loaded from saved baseline results)
- LightGBM point forecast
- LightGBM quantile forecast
- XGBoost point forecast
- Prophet on a small sample of high-volume series
- SARIMAX on a small sample of high-volume series

This script saves:
- forecasting_results.csv / .json
- model artifacts
- feature importance plots
- quantile calibration plot
- one sample forecast plot with prediction intervals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
from src.forecasting.prophet_forecaster import run_prophet_on_sample
from src.forecasting.sarimax_forecaster import run_sarimax_on_sample
from src.forecasting.xgb_forecaster import XGBPointForecaster
from src.utils.logger import get_logger


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "forecasting",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]
    models_dir = PROJECT_ROOT / config["outputs"]["models_dir"]

    for directory in [results_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for split in ["train_features", "val_features", "test_features"]:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            logger.error("%s not found. Run scripts/feature_engineering.py first.", path)
            sys.exit(1)

    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    val_df = pd.read_parquet(processed_dir / "val_features.parquet")
    test_df = pd.read_parquet(processed_dir / "test_features.parquet")

    target_col = config["data"]["target_column"]
    date_col = config["data"]["date_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    promo_col = config["data"]["promo_column"]

    full_df = (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values([store_col, item_col, date_col])
        .reset_index(drop=True)
    )

    if promo_col in full_df.columns:
        full_df = add_promotion_features(
            full_df,
            promo_col=promo_col,
            group_cols=[store_col, item_col],
            date_col=date_col,
        )

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

    feature_cols = get_feature_columns(train_df, config)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col].values

    logger.info(
        "Feature matrix | train=%s | val=%s | test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    all_results: list[dict] = []

    baseline_path = results_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

        for row in baseline_results:
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

    logger.info("Training LightGBM point forecast")
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

    logger.info("Training LightGBM quantile forecast")
    lgbm_q = LGBMQuantileForecaster(quantiles=[0.05, 0.5, 0.95])
    lgbm_q.fit(X_train, y_train, X_val, y_val)
    lgbm_q.save(models_dir / "lgbm_quantile")

    q_preds = lgbm_q.predict(X_test)

    all_results.append(
        evaluate_probabilistic_forecast(
            y_test,
            q_preds["q0.05"],
            q_preds["q0.5"],
            q_preds["q0.95"],
            "LightGBM Quantile (Q0.05/0.5/0.95)",
        )
    )

    q_preds_df = pd.DataFrame(
        {
            date_col: test_df[date_col].values,
            store_col: test_df[store_col].values,
            item_col: test_df[item_col].values,
            "actual": y_test,
            "q0.05": q_preds["q0.05"],
            "q0.5": q_preds["q0.5"],
            "q0.95": q_preds["q0.95"],
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
            y_pred_q05=q_preds["q0.05"][sample_mask.values],
            y_pred_q50=q_preds["q0.5"][sample_mask.values],
            y_pred_q95=q_preds["q0.95"][sample_mask.values],
            title=f"Demand Forecast with 90% PI — Store {first_store}, Item {first_item}",
            save_path=figures_dir / "sample_quantile_forecast.png",
        )

    logger.info("Training XGBoost point forecast")
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

    logger.info("Running Prophet on a sample")
    prophet_df = run_prophet_on_sample(train_df, test_df, config, n_series=10)
    if not prophet_df.empty:
        prophet_df.to_csv(results_dir / "prophet_predictions.csv", index=False)

        prophet_result = evaluate_point_forecast(
            prophet_df[target_col].values,
            prophet_df["yhat"].values,
            "Prophet (10-series sample)",
        )
        prophet_result["coverage_90"] = round(
            float(
                np.mean(
                    (prophet_df[target_col].values >= prophet_df["yhat_lower"].values) &
                    (prophet_df[target_col].values <= prophet_df["yhat_upper"].values)
                )
            ),
            4,
        )
        prophet_result["interval_width"] = round(
            float(np.mean(prophet_df["yhat_upper"].values - prophet_df["yhat_lower"].values)),
            4,
        )
        all_results.append(prophet_result)

    logger.info("Running SARIMAX on a sample")
    sarimax_df = run_sarimax_on_sample(train_df, test_df, config, n_series=20)
    if not sarimax_df.empty:
        sarimax_df.to_csv(results_dir / "sarimax_predictions.csv", index=False)

        sarimax_result = evaluate_point_forecast(
            sarimax_df[target_col].values,
            sarimax_df["forecast"].values,
            "SARIMAX (20-series sample)",
        )
        sarimax_result["coverage_90"] = round(
            float(
                np.mean(
                    (sarimax_df[target_col].values >= sarimax_df["lower_ci"].values) &
                    (sarimax_df[target_col].values <= sarimax_df["upper_ci"].values)
                )
            ),
            4,
        )
        sarimax_result["interval_width"] = round(
            float(np.mean(sarimax_df["upper_ci"].values - sarimax_df["lower_ci"].values)),
            4,
        )
        all_results.append(sarimax_result)

    results_df = build_results_table(all_results)
    save_results(results_df, results_dir, name="forecasting_results")

    logger.info("Forecasting pipeline complete")


if __name__ == "__main__":
    main()