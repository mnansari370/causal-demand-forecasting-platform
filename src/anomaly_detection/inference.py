"""
Run anomaly detection on real demand series.

This is separate from the held-out synthetic test evaluation. Here we render
real time-series from the demand dataset into chart images and pass them
through the trained detector.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _render_series_to_image(series: np.ndarray, image_size: int = 224) -> Image.Image:
    """
    Render a one-dimensional series into a chart image.
    """
    dpi = 100
    fig_size = image_size / dpi

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.plot(series, linewidth=1.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    plt.tight_layout(pad=0)

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize((image_size, image_size))


def _get_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def run_anomaly_inference(
    test_df: pd.DataFrame,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    target_col: str,
    store_col: str,
    item_col: str,
    date_col: str,
    top_n: int = 50,
    figures_dir: str | Path | None = None,
    results_path: str | Path | None = None,
) -> list[dict]:
    """
    Run the trained detector on the top-N real series by total demand.
    """
    transform = _get_transform(image_size=image_size)

    top_series = (
        test_df.groupby([store_col, item_col])[target_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    logger.info("Running anomaly inference on %d real series", len(top_series))

    model.eval()
    results: list[dict] = []

    figures_dir = Path(figures_dir) if figures_dir is not None else None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    for idx, (store, item) in enumerate(top_series, start=1):
        series = (
            test_df[
                (test_df[store_col] == store) &
                (test_df[item_col] == item)
            ]
            .sort_values(date_col)[target_col]
            .values.astype(float)
        )

        if len(series) < 5:
            continue

        pil_img = _render_series_to_image(series, image_size=image_size)
        image_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(image_tensor), dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        is_anomaly = pred_class != "normal"

        row = {
            store_col: int(store),
            item_col: int(item),
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "is_anomaly": bool(is_anomaly),
            "class_probabilities": {
                cls: round(float(probs[i]), 4)
                for i, cls in enumerate(class_names)
            },
            "mean_sales": round(float(np.mean(series)), 4),
            "series_length": int(len(series)),
        }
        results.append(row)

        if is_anomaly:
            logger.info(
                "[%d/%d] store=%s item=%s -> %s (conf=%.4f) ANOMALY",
                idx,
                len(top_series),
                store,
                item,
                pred_class,
                confidence,
            )

            if figures_dir is not None:
                save_path = figures_dir / f"flagged_{pred_class}_s{store}_i{item}.png"
                pil_img.save(save_path)
        else:
            logger.info(
                "[%d/%d] store=%s item=%s -> normal (conf=%.4f)",
                idx,
                len(top_series),
                store,
                item,
                confidence,
            )

    anomaly_count = sum(r["is_anomaly"] for r in results)
    logger.info("Inference complete | total=%d | anomalies=%d", len(results), anomaly_count)

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved inference results: %s", results_path)

    return results