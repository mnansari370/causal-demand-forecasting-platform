"""
Evaluate the anomaly detector on the held-out synthetic test split.

This module also includes Grad-CAM visualisation to check whether the
model focuses on the visually meaningful region of the chart.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.anomaly_detection.split_dataset import load_split_manifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticChartDataset(Dataset):
    """
    Dataset for the held-out synthetic test split.
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _get_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_trained_model(
    model_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], int]:
    """
    Load a saved ResNet-18 checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names)),
    )

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded detector from %s | val_acc=%.4f | classes=%s",
        model_path,
        checkpoint["val_acc"],
        class_names,
    )

    return model, class_names, image_size


def evaluate_on_test_split(
    model: nn.Module,
    manifest_path: str | Path,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate the trained detector on the held-out synthetic test set.
    """
    manifest = load_split_manifest(manifest_path)

    class_names = sorted(manifest["splits"]["test"].keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    test_samples = [
        (path, class_to_idx[cls])
        for cls in class_names
        for path in manifest["splits"]["test"][cls]
    ]

    loader = DataLoader(
        SyntheticChartDataset(test_samples, _get_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1

    per_class = {}

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        per_class[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(cm[i, :].sum()),
        }

    accuracy = float(np.mean(all_preds == all_labels))
    macro_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()]))

    result = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_test_samples": int(len(all_labels)),
    }

    logger.info("Test evaluation | accuracy=%.4f | macro_f1=%.4f", accuracy, macro_f1)
    return result


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """
    Save a normalised confusion matrix plot.
    """
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalised)")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({int(cm_arr[i, j])})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved confusion matrix: %s", save_path)


def plot_per_class_metrics(
    per_class: dict,
    save_path: str | Path,
) -> None:
    """
    Save a grouped bar chart for precision, recall, and F1.
    """
    classes = list(per_class.keys())
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [per_class[c]["precision"] for c in classes], width, label="Precision")
    ax.bar(x, [per_class[c]["recall"] for c in classes], width, label="Recall")
    ax.bar(x + width, [per_class[c]["f1"] for c in classes], width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved per-class metrics plot: %s", save_path)


class GradCAM:
    """
    Simple Grad-CAM implementation for ResNet-18.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.layer4[-1]
        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3))[0]
        cam = sum(w * self.activations[0, i] for i, w in enumerate(weights))
        cam = F.relu(cam).unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_gradcam_examples(
    model: nn.Module,
    manifest_path: str | Path,
    image_size: int,
    device: torch.device,
    figures_dir: str | Path,
    n_per_class: int = 3,
) -> None:
    """
    Generate Grad-CAM examples for correctly classified anomaly classes.
    """
    manifest = load_split_manifest(manifest_path)
    class_names = sorted(manifest["splits"]["test"].keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    tf = _get_eval_transform(image_size)
    gradcam = GradCAM(model)
    figures_dir = Path(figures_dir)

    for cls_name in [c for c in class_names if c != "normal"]:
        cls_idx = class_to_idx[cls_name]
        saved = 0

        for img_path in manifest["splits"]["test"][cls_name]:
            if saved >= n_per_class:
                break

            pil_img = Image.open(img_path).convert("RGB")
            tensor = tf(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())

            if pred_idx != cls_idx:
                continue

            cam = gradcam.generate(tensor, class_idx=pred_idx)

            display_img = np.array(pil_img.resize((image_size, image_size))).astype(np.float32) / 255.0
            heatmap = plt.cm.jet(cam)[..., :3]
            overlay = np.clip(0.6 * display_img + 0.4 * heatmap, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(display_img)
            axes[0].axis("off")
            axes[0].set_title("Original")

            axes[1].imshow(cam, cmap="jet")
            axes[1].axis("off")
            axes[1].set_title("Grad-CAM")

            axes[2].imshow(overlay)
            axes[2].axis("off")
            axes[2].set_title(f"{cls_name} (conf={confidence:.2f})")

            plt.tight_layout()

            save_path = figures_dir / f"gradcam_{cls_name}_{saved + 1}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved Grad-CAM example: %s", save_path)
            saved += 1

        if saved == 0:
            logger.warning("No correctly classified examples found for class: %s", cls_name)

    gradcam.remove_hooks()