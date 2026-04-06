from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.evaluation.metrics import coverage_at_90, mae, mape, rmse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_point_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    result = {
        "model": model_name,
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
        "coverage_90": None,
        "interval_width": None,
        "n_samples": int(len(y_true)),
    }

    logger.info(
        "%s | RMSE=%.4f | MAE=%.4f | MAPE=%.2f%%",
        model_name,
        result["rmse"],
        result["mae"],
        result["mape"],
    )
    return result


def evaluate_probabilistic_forecast(
    y_true: np.ndarray,
    y_pred_q10: np.ndarray,
    y_pred_q50: np.ndarray,
    y_pred_q90: np.ndarray,
    model_name: str,
) -> dict:
    result = evaluate_point_forecast(y_true, y_pred_q50, model_name)

    cov = coverage_at_90(y_true, y_pred_q10, y_pred_q90)
    width = float(np.mean(y_pred_q90 - y_pred_q10))

    result["coverage_90"] = round(cov, 4)
    result["interval_width"] = round(width, 4)

    logger.info(
        "%s | Coverage@90=%.4f | Interval Width=%.4f",
        model_name,
        result["coverage_90"],
        result["interval_width"],
    )
    return result


def build_results_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    col_order = [
        "model",
        "rmse",
        "mae",
        "mape",
        "coverage_90",
        "interval_width",
        "n_samples",
    ]
    existing = [c for c in col_order if c in df.columns]
    return df[existing].reset_index(drop=True)


def save_results(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    name: str = "forecasting_results",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"

    results_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results_df.to_dict(orient="records"), indent=2))

    logger.info("Saved results to: %s", output_dir)

    print("\n" + "=" * 80)
    print("WEEK 2 FORECASTING RESULTS — TEST SET")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print()

    baseline_rows = results_df[results_df["model"] == "Seasonal Naive (S=7)"]
    if not baseline_rows.empty:
        b = baseline_rows.iloc[0]
        print(
            f"Baseline to beat: RMSE={b['rmse']:.4f} | MAE={b['mae']:.4f} "
            "(Seasonal Naive, test set)"
        )
        print()


def plot_forecast_with_intervals(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred_q10: np.ndarray,
    y_pred_q50: np.ndarray,
    y_pred_q90: np.ndarray,
    title: str = "Demand Forecast with 90% Prediction Interval",
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, y_true, label="Actual", linewidth=1.5, zorder=3)
    ax.plot(dates, y_pred_q50, label="Forecast (Q0.5)", linewidth=1.5, linestyle="--", zorder=3)
    ax.fill_between(
        dates,
        y_pred_q10,
        y_pred_q90,
        alpha=0.25,
        label="90% Prediction Interval",
        zorder=2,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved forecast interval plot: %s", save_path)

    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 25,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> None:
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot: %s", save_path)

    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    quantile_preds: dict[str, np.ndarray],
    title: str = "Quantile Calibration",
    save_path: str | Path | None = None,
) -> None:
    quantile_keys = sorted(quantile_preds.keys())
    nominal = []
    observed = []

    for key in quantile_keys:
        q_val = float(key.replace("q", ""))
        frac = float(np.mean(y_true <= quantile_preds[key]))
        nominal.append(q_val)
        observed.append(frac)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(nominal, observed, s=80, zorder=3)
    ax.plot(nominal, observed, linewidth=1.5, label="Model")
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Observed fraction below quantile")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved calibration plot: %s", save_path)

    plt.close()