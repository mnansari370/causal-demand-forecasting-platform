"""
Plotting utilities for causal inference results.
"""
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
    save_path: str | Path,
) -> None:
    """
    Plot naive estimate, DiD estimate, and placebo estimate side by side.

    This visualises:
    - how biased the naive estimate is,
    - the causal estimate from DiD,
    - whether the placebo result stays close to zero.
    """
    naive_est = naive_result.get("naive_estimate", 0) or 0
    did_est = did_result.get("estimate", 0) or 0
    placebo_est = placebo_result.get("estimate", 0) or 0

    labels = ["Naive\n(biased)", "DiD\n(causal)", "Placebo\n(~0 expected)"]
    values = [naive_est, did_est, placebo_est]
    colors = ["#e74c3c", "#27ae60", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(3), values, color=colors, alpha=0.85)

    for i, val in enumerate(values):
        ax.text(i, val + 0.05, f"{val:+.2f}", ha="center", fontsize=10)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Promotion effect (units/day)")
    ax.set_title("Naive vs Causal (DiD) vs Placebo Estimate")

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_hte_ranking(
    hte_df: pd.DataFrame,
    id_col: str,
    estimate_col: str = "promotion_lift_estimate",
    top_n: int = 10,
    title: str = "HTE Ranking",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the top and bottom heterogeneous treatment effects in one chart.
    """
    if hte_df.empty:
        logger.warning("HTE dataframe is empty — skipping ranking plot")
        return

    plot_df = pd.concat([hte_df.head(top_n), hte_df.tail(top_n)]).sort_values(estimate_col)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
    ax.barh(y, plot_df[estimate_col].values, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[id_col].astype(str))
    ax.set_xlabel("Promotion lift (units/day)")
    ax.set_title(title)
    ax.axvline(0, linestyle="--", linewidth=1)

    plt.tight_layout()

    if save_path is not None:
        _save_fig(fig, save_path)
    else:
        plt.close(fig)


def _save_fig(fig, path: str | Path) -> None:
    """
    Save and close a matplotlib figure cleanly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)