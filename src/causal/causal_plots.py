from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_did_summary(
    did_result: dict,
    placebo_result: dict,
    naive_result: dict,
    save_path: str | Path | None = None,
) -> None:
    labels = ["Naive estimate\n(biased)", "DiD estimate\n(causal)", "Placebo test\n(should be ~0)"]
    estimates = [
        naive_result.get("naive_estimate", 0),
        did_result.get("estimate", 0),
        placebo_result.get("estimate", 0) if placebo_result.get("estimate") is not None else 0,
    ]
    ci_low = [
        np.nan,
        did_result.get("ci_low", np.nan),
        placebo_result.get("ci_low", np.nan),
    ]
    ci_high = [
        np.nan,
        did_result.get("ci_high", np.nan),
        placebo_result.get("ci_high", np.nan),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, estimates, width=0.5, alpha=0.85, zorder=3)

    for i in range(1, len(estimates)):
        if not np.isnan(ci_low[i]) and not np.isnan(ci_high[i]):
            ax.errorbar(
                x[i],
                estimates[i],
                yerr=[[estimates[i] - ci_low[i]], [ci_high[i] - estimates[i]]],
                fmt="none",
                color="black",
                capsize=6,
                linewidth=2,
                zorder=4,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Estimated promotion lift (unit sales)", fontsize=11)
    ax.set_title("Naive vs Causal (DiD) Promotion Effect Estimates", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, estimates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("DiD summary chart saved: %s", save_path)

    plt.close()


def plot_hte_table(
    hte_df: pd.DataFrame,
    id_col: str,
    estimate_col: str = "promotion_lift_estimate",
    ci_lower_col: str = "te_lower",
    ci_upper_col: str = "te_upper",
    top_n: int = 15,
    title: str = "Promotion Sensitivity by Entity",
    save_path: str | Path | None = None,
) -> None:
    top = hte_df.head(top_n).copy()
    bottom = hte_df.tail(top_n).copy()
    plot_df = pd.concat([top, bottom], ignore_index=True)
    plot_df = plot_df.sort_values(estimate_col, ascending=True)

    y_pos = np.arange(len(plot_df))
    estimates = plot_df[estimate_col].values
    errors_lo = estimates - plot_df[ci_lower_col].values
    errors_hi = plot_df[ci_upper_col].values - estimates

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))

    ax.barh(y_pos, estimates, alpha=0.8, height=0.6)
    ax.errorbar(
        estimates,
        y_pos,
        xerr=[errors_lo, errors_hi],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.2,
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df[id_col].astype(str), fontsize=9)
    ax.set_xlabel("Estimated promotion lift (unit sales)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("HTE chart saved: %s", save_path)

    plt.close()