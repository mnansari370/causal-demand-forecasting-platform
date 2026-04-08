from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.utils.logger import get_logger

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logger = get_logger(__name__)


def _base_series(length: int = 90) -> np.ndarray:
    t = np.arange(length, dtype=float)
    base = random.uniform(5.0, 100.0)
    trend = random.uniform(-0.08, 0.12) * t
    seasonality = random.uniform(0.5, 2.5) * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(0, random.uniform(0.5, 4.0), length)
    return np.maximum(base + trend + seasonality + noise, 0.0)


def _inject_spike(series: np.ndarray, pos: int) -> np.ndarray:
    s = series.copy()
    local_mean = np.mean(s[max(0, pos - 7):pos]) + 1e-3
    s[pos] = local_mean * random.uniform(2.5, 6.0)
    return s


def _inject_drop(series: np.ndarray, pos: int) -> np.ndarray:
    s = series.copy()
    local_mean = np.mean(s[max(0, pos - 7):pos]) + 1e-3
    duration = random.randint(3, 10)
    end = min(pos + duration, len(s))
    s[pos:end] = local_mean * random.uniform(0.0, 0.2)
    return s


def _inject_structural_break(series: np.ndarray, pos: int) -> np.ndarray:
    s = series.copy()
    before_mean = np.mean(s[:pos]) + 1e-3
    shift = before_mean * random.uniform(0.3, 0.7)
    if random.random() < 0.5:
        s[pos:] = np.maximum(s[pos:] - shift, 0.0)
    else:
        s[pos:] = s[pos:] + shift
    return s


def _render_chart(series: np.ndarray, save_path: Path, image_size: int = 224) -> None:
    dpi = 100
    fig_size = image_size / dpi

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.plot(series, linewidth=1.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_anomaly_dataset(
    output_dir: Path,
    n_per_class: int = 2000,
    image_size: int = 224,
    series_length: int = 90,
) -> None:
    classes = ["normal", "spike", "drop", "structural_break"]

    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    total = n_per_class * len(classes)
    logger.info("Generating %d images (%d per class) -> %s", total, n_per_class, output_dir)

    counts: dict[str, int] = {cls: 0 for cls in classes}

    for cls in classes:
        logger.info("Generating class: %s", cls)

        for i in tqdm(range(n_per_class), desc=cls):
            base = _base_series(series_length)
            pos = random.randint(10, series_length - 15)

            if cls == "normal":
                series = base
            elif cls == "spike":
                series = _inject_spike(base, pos)
            elif cls == "drop":
                series = _inject_drop(base, pos)
            else:
                series = _inject_structural_break(base, pos)

            img_path = output_dir / cls / f"{cls}_{i:05d}.png"
            _render_chart(series, img_path, image_size=image_size)
            counts[cls] += 1

    logger.info("Dataset generation complete:")
    for cls, count in counts.items():
        logger.info("  %-18s %d images", cls, count)
    logger.info("Total: %d images", sum(counts.values()))


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    cv_cfg = config["cv_anomaly"]

    generate_anomaly_dataset(
        output_dir=PROJECT_ROOT / cv_cfg["synthetic_data_dir"],
        n_per_class=cv_cfg["n_images_per_class"],
        image_size=cv_cfg["image_size"],
        series_length=90,
    )