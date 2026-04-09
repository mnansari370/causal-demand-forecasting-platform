from __future__ import annotations

import json
import sys
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
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_trained_model(
    model_path: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    n_classes = len(class_names)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded anomaly detector from %s | classes=%s",
        model_path,
        class_names,
    )

    return model, class_names, image_size


def render_series_chart(series: np.ndarray, image_size: int = 224) -> Image.Image:
    """
    Render a single time-series to a chart image consistent with synthetic training data.

    Uses an in-memory PNG buffer instead of canvas.tostring_rgb(), which is more
    robust across matplotlib backend/version differences.
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
    image = Image.open(buffer).convert("RGB")
    image = image.resize((image_size, image_size))
    return image


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
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
    figures_dir: Path | None = None,
    results_path: Path | None = None,
) -> list[dict]:
    """
    Run anomaly detector on top-N real test-set series by total demand.

    This is deployment-style qualitative inference, not the held-out synthetic test evaluation.
    """
    tf = get_inference_transform(image_size=image_size)

    top_series = (
        test_df.groupby([store_col, item_col])[target_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    logger.info("Running real-series anomaly inference on %d series", len(top_series))

    results: list[dict] = []
    model.eval()

    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    for idx, (store, item) in enumerate(top_series, start=1):
        series_df = test_df[
            (test_df[store_col] == store) & (test_df[item_col] == item)
        ].sort_values(date_col)

        sales = series_df[target_col].values.astype(float)

        if len(sales) < 5:
            logger.info(
                "Skipping short series | store=%s item=%s length=%d",
                store, item, len(sales)
            )
            continue

        pil_img = render_series_chart(sales, image_size=image_size)
        image_tensor = tf(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

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
            "mean_sales": round(float(np.mean(sales)), 4),
            "max_sales": round(float(np.max(sales)), 4),
            "min_sales": round(float(np.min(sales)), 4),
            "series_length": int(len(sales)),
        }
        results.append(row)

        if is_anomaly:
            logger.info(
                "[%d/%d] store=%s item=%s -> %s (conf=%.4f) ANOMALY",
                idx, len(top_series), store, item, pred_class, confidence
            )
            if figures_dir is not None:
                save_path = figures_dir / f"flagged_{pred_class}_store{store}_item{item}.png"
                pil_img.save(save_path)
        else:
            logger.info(
                "[%d/%d] store=%s item=%s -> normal (conf=%.4f)",
                idx, len(top_series), store, item, confidence
            )

    anomaly_count = sum(r["is_anomaly"] for r in results)
    logger.info(
        "Real-series inference complete | total=%d | anomalies=%d",
        len(results),
        anomaly_count,
    )

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved real-series anomaly results: %s", results_path)

    return results


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]
    model_path = PROJECT_ROOT / config["cv_anomaly"]["model_save_path"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"] / "flagged_charts"
    results_path = PROJECT_ROOT / config["evaluation"]["results_dir"] / "anomaly_detection_results.json"

    test_path = processed_dir / "test_features.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test features parquet: {test_path}")

    test_df = pd.read_parquet(test_path)

    target_col = config["data"]["target_column"]
    store_col = config["data"]["store_column"]
    item_col = config["data"]["item_column"]
    date_col = config["data"]["date_column"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_trained_model(model_path, device)

    run_anomaly_inference(
        test_df=test_df,
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
        target_col=target_col,
        store_col=store_col,
        item_col=item_col,
        date_col=date_col,
        top_n=50,
        figures_dir=figures_dir,
        results_path=results_path,
    )