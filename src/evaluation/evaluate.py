"""
Evaluation utilities: metric computation, result tables, and plots.

All figures are saved to disk using the Agg backend so the code works
both locally and on HPC without any display.
"""
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
    """
    Evaluate a point forecast using RMSE, MAE, and MAPE.
    """
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
    y_pred_q05: np.ndarray,
    y_pred_q50: np.ndarray,
    y_pred_q95: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a probabilistic forecast using median accuracy, coverage, and interval width.
    """
    result = evaluate_point_forecast(y_true, y_pred_q50, model_name)

    cov = coverage_at_90(y_true, y_pred_q05, y_pred_q95)
    width = float(np.mean(y_pred_q95 - y_pred_q05))

    result["coverage_90"] = round(cov, 4)
    result["interval_width"] = round(width, 4)

    logger.info(
        "%s | Coverage@90=%.4f | IntervalWidth=%.4f",
        model_name,
        cov,
        width,
    )
    return result


def build_results_table(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of metric dictionaries into a standard results table.
    """
    col_order = [
        "model",
        "rmse",
        "mae",
        "mape",
        "coverage_90",
        "interval_width",
        "n_samples",
    ]

    df = pd.DataFrame(results)
    existing = [c for c in col_order if c in df.columns]
    return df[existing].reset_index(drop=True)


def save_results(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    name: str = "forecasting_results",
) -> None:
    """
    Save results as both CSV and JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"

    results_df.to_csv(csv_path, index=False)

    records = results_df.where(results_df.notna(), other=None).to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2))

    logger.info("Saved results to: %s", output_dir)

    print("\n" + "=" * 80)
    print("FORECASTING RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)


def plot_forecast_with_intervals(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred_q05: np.ndarray,
    y_pred_q50: np.ndarray,
    y_pred_q95: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """
    Plot actual values, median forecast, and 90% prediction interval.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, y_true, label="Actual", linewidth=1.5)
    ax.plot(dates, y_pred_q50, label="Forecast (median)", linewidth=1.5, linestyle="--")
    ax.fill_between(dates, y_pred_q05, y_pred_q95, alpha=0.25, label="90% PI")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 25,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot horizontal bar chart of top feature importances.
    """
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1])

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.close()


def plot_calibration(
    y_true: np.ndarray,
    quantile_preds: dict[str, np.ndarray],
    title: str = "Quantile Calibration",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot observed coverage against nominal quantiles.

    A perfectly calibrated model lies on the diagonal.
    """
    quantile_keys = sorted(quantile_preds.keys())
    nominal = []
    observed = []

    for key in quantile_keys:
        q_val = float(key.replace("q", ""))
        nominal.append(q_val)
        observed.append(float(np.mean(y_true <= quantile_preds[key])))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(nominal, observed, s=80, zorder=3)
    ax.plot(nominal, observed, linewidth=1.5, label="Model")

    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Observed fraction below quantile")
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.close()


def _save_fig(fig, path: str | Path) -> None:
    """
    Save a matplotlib figure and close it cleanly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)