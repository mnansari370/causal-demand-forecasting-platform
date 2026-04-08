from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: must be before pyplot

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
    """
    Compare naive vs DiD vs placebo estimates.
    """

    naive_est = naive_result.get("naive_estimate", 0) or 0
    did_est = did_result.get("estimate", 0) or 0
    placebo_est = placebo_result.get("estimate", 0) or 0

    labels = ["Naive\n(biased)", "DiD\n(causal)", "Placebo\n(≈0)"]
    values = [naive_est, did_est, placebo_est]
    colors = ["#e74c3c", "#27ae60", "#95a5a6"]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, values, color=colors, alpha=0.85)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:+.2f}", ha="center", fontsize=10)

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Promotion effect (units/day)")
    ax.set_title("Naive vs Causal Effect (DiD) vs Placebo")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved DiD summary plot: %s", save_path)

    plt.close()


def plot_hte_ranking(
    hte_df: pd.DataFrame,
    id_col: str,
    estimate_col: str = "promotion_lift_estimate",
    ci_lower_col: str = "te_lower",
    ci_upper_col: str = "te_upper",
    top_n: int = 10,
    title: str = "HTE Ranking",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot top-N and bottom-N treatment effects.
    """

    if len(hte_df) == 0:
        logger.warning("Empty HTE DataFrame — skipping plot")
        return

    top = hte_df.head(top_n)
    bottom = hte_df.tail(top_n)

    plot_df = pd.concat([top, bottom]).sort_values(estimate_col)

    y = np.arange(len(plot_df))
    values = plot_df[estimate_col].values

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))

    ax.barh(y, values, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[id_col].astype(str))
    ax.set_xlabel("Promotion Lift (units/day)")
    ax.set_title(title)

    ax.axvline(0, linestyle="--", linewidth=1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved HTE plot: %s", save_path)

    plt.close()