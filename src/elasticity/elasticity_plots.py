from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_family_promotion_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> None:
    if sensitivity_df.empty:
        logger.warning("Empty sensitivity DataFrame; skipping family sensitivity plot")
        return

    if len(sensitivity_df) > top_n:
        top_half = sensitivity_df.nlargest(top_n // 2, "promotion_coef")
        bottom_half = sensitivity_df.nsmallest(top_n // 2, "promotion_coef")
        plot_df = pd.concat([top_half, bottom_half], ignore_index=True)
    else:
        plot_df = sensitivity_df.copy()

    plot_df = plot_df.sort_values("promotion_coef", ascending=True).reset_index(drop=True)

    colors = ["#27ae60" if sig else "#95a5a6" for sig in plot_df["significant"]]

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.4)))
    y = np.arange(len(plot_df))

    ax.barh(y, plot_df["promotion_coef"], color=colors, alpha=0.85)

    err_low = plot_df["promotion_coef"] - plot_df["ci_low"]
    err_high = plot_df["ci_high"] - plot_df["promotion_coef"]

    ax.errorbar(
        plot_df["promotion_coef"],
        y,
        xerr=[err_low, err_high],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["family"].astype(str))
    ax.set_xlabel("Promotion coefficient")
    ax.set_title("Promotion Sensitivity by Family")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved family promotion sensitivity plot: %s", save_path)

    plt.close()


def plot_revenue_proxy_curve(
    revenue_df: pd.DataFrame,
    label_name: str = "Selected Group",
    save_path: str | Path | None = None,
) -> None:
    if revenue_df.empty:
        logger.warning("Empty revenue proxy DataFrame; skipping plot")
        return

    labels = ["Promotion OFF" if x == 0 else "Promotion ON" for x in revenue_df["promotion_on"]]
    values = revenue_df["revenue_proxy"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Revenue Proxy")
    ax.set_title(f"Revenue Proxy Comparison — {label_name}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved revenue proxy plot: %s", save_path)

    plt.close()


def plot_scenario_comparison(
    scenario_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    if scenario_df.empty:
        logger.warning("Empty scenario DataFrame; skipping plot")
        return

    plot_df = scenario_df.copy()
    plot_df["scenario"] = plot_df["run_promotion"].map({False: "No Promotion", True: "Promotion"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        plot_df["scenario"],
        plot_df["revenue_delta_pct"],
        alpha=0.85,
    )

    for i, val in enumerate(plot_df["revenue_delta_pct"]):
        ax.text(i, val, f"{val:+.2f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Revenue Delta (%)")
    ax.set_title("Scenario Comparison — Revenue Delta vs Baseline")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved scenario comparison plot: %s", save_path)

    plt.close()


def plot_simulation_output(
    scenario: dict,
    save_path: str | Path | None = None,
) -> None:
    labels = ["Q0.05", "Q0.50", "Q0.95"]
    values = [
        scenario["expected_revenue_q05"],
        scenario["expected_revenue_q50"],
        scenario["expected_revenue_q95"],
    ]
    baseline = scenario["baseline_revenue"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, alpha=0.85)
    ax.axhline(baseline, linestyle="--", linewidth=1.5, label=f"Baseline={baseline:.2f}")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Revenue Proxy")
    ax.set_title(
        f"Simulation Output | Promotion={'YES' if scenario['run_promotion'] else 'NO'} | "
        f"Delta={scenario['revenue_delta_pct']:+.2f}%"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved simulation output plot: %s", save_path)

    plt.close()
