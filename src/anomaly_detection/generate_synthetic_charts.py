"""
Synthetic chart generator for visual anomaly detection.

We generate labelled chart images because real labelled anomaly data is not
available at scale. The goal is to teach the model the visual patterns of:

- normal demand
- spike anomaly
- drop anomaly
- structural break

These synthetic charts are later used to train the ResNet-based detector.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _generate_base_series(length: int = 90) -> np.ndarray:
    """
    Create a smooth retail-like demand series with trend, seasonality, and noise.
    """
    x = np.arange(length)

    level = random.uniform(20, 100)
    trend = random.uniform(-0.2, 0.2) * x
    seasonality = 8.0 * np.sin(2 * np.pi * x / 7.0)
    noise = np.random.normal(0, random.uniform(1.0, 4.0), size=length)

    y = level + trend + seasonality + noise
    return np.maximum(y, 0.0)


def _add_spike(series: np.ndarray) -> np.ndarray:
    """
    Add one large temporary spike.
    """
    y = series.copy()
    idx = random.randint(10, len(y) - 10)
    y[idx] = y[idx] * random.uniform(2.5, 6.0)
    return y


def _add_drop(series: np.ndarray) -> np.ndarray:
    """
    Add a temporary sustained drop.
    """
    y = series.copy()
    start = random.randint(10, len(y) - 15)
    duration = random.randint(3, 10)
    end = min(start + duration, len(y))
    y[start:end] = y[start:end] * random.uniform(0.0, 0.2)
    return y


def _add_structural_break(series: np.ndarray) -> np.ndarray:
    """
    Add a persistent level shift upward or downward.
    """
    y = series.copy()
    idx = random.randint(15, len(y) - 15)
    shift = np.mean(y[:idx]) * random.uniform(0.3, 0.7)

    if random.random() < 0.5:
        y[idx:] = np.maximum(y[idx:] - shift, 0.0)
    else:
        y[idx:] = y[idx:] + shift

    return y


def _save_chart(series: np.ndarray, save_path: Path, image_size: int = 224) -> None:
    """
    Render one time-series as a clean chart image.
    """
    dpi = 100
    fig_size = image_size / dpi

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.plot(series, linewidth=1.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    plt.tight_layout(pad=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_synthetic_chart_dataset(
    output_dir: str | Path,
    n_per_class: int = 2000,
    image_size: int = 224,
    series_length: int = 90,
) -> None:
    """
    Generate synthetic chart images for all anomaly classes.
    """
    output_dir = Path(output_dir)
    classes = ["normal", "spike", "drop", "structural_break"]

    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating synthetic anomaly dataset | classes=%s | n_per_class=%d",
        classes,
        n_per_class,
    )

    for cls in classes:
        logger.info("Generating class: %s", cls)

        for i in range(n_per_class):
            base = _generate_base_series(length=series_length)

            if cls == "normal":
                series = base
            elif cls == "spike":
                series = _add_spike(base)
            elif cls == "drop":
                series = _add_drop(base)
            else:
                series = _add_structural_break(base)

            save_path = output_dir / cls / f"{cls}_{i:05d}.png"
            _save_chart(series, save_path, image_size=image_size)

    logger.info("Synthetic anomaly dataset generation complete: %s", output_dir)


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))

    from src.data.load_data import load_config

    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    anomaly_cfg = config["cv_anomaly"]

    output_dir = PROJECT_ROOT / anomaly_cfg["synthetic_data_dir"]
    n_per_class = anomaly_cfg.get("n_images_per_class", 2000)
    image_size = anomaly_cfg.get("image_size", 224)
    series_length = anomaly_cfg.get("series_length", 90)

    generate_synthetic_chart_dataset(
        output_dir=output_dir,
        n_per_class=n_per_class,
        image_size=image_size,
        series_length=series_length,
    )


if __name__ == "__main__":
    main()