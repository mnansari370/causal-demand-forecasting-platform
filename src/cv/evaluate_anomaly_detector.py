from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.cv.split_dataset import load_split_manifest
from src.data.load_data import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticChartDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


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
        "Loaded trained model: %s | val_acc=%.4f | classes=%s",
        model_path,
        checkpoint["val_acc"],
        class_names,
    )

    return model, class_names, image_size


def build_test_loader_from_manifest(
    manifest_path: Path,
    image_size: int,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> tuple[DataLoader, list[str], list[tuple[str, int]]]:
    manifest = load_split_manifest(manifest_path)
    class_names = sorted(manifest["splits"]["test"].keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    test_samples: list[tuple[str, int]] = []
    for cls in class_names:
        test_samples.extend(
            [(path, class_to_idx[cls]) for path in manifest["splits"]["test"][cls]]
        )

    test_ds = SyntheticChartDataset(
        test_samples,
        transform=get_eval_transform(image_size=image_size),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info("Synthetic CV test set ready | size=%d", len(test_ds))
    return test_loader, class_names, test_samples


def evaluate_on_test_split(
    model: nn.Module,
    manifest_path: Path,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    test_loader, class_names, _ = build_test_loader_from_manifest(
        manifest_path=manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1

    per_class_metrics: dict[str, dict] = {}

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

        per_class_metrics[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(cm[i, :].sum()),
        }

    macro_f1 = float(np.mean([v["f1"] for v in per_class_metrics.values()]))
    accuracy = float(np.mean(all_preds == all_labels))

    result = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_test_samples": int(len(all_labels)),
    }

    logger.info(
        "CV synthetic test evaluation complete | accuracy=%.4f | macro_f1=%.4f",
        accuracy,
        macro_f1,
    )

    return result


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    save_path: Path,
) -> None:
    cm_arr = np.array(cm)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_arr.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of true class")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CV Confusion Matrix (Normalised)")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm_arr[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved confusion matrix: %s", save_path)


def plot_per_class_f1(
    per_class: dict,
    save_path: Path,
) -> None:
    classes = list(per_class.keys())
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    f1s = [per_class[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precisions, w, label="Precision")
    ax.bar(x, recalls, w, label="Recall")
    ax.bar(x + w, f1s, w, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CV Precision / Recall / F1 Per Class")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved per-class metric plot: %s", save_path)


class GradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.layer4[-1]
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3))[0]
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = F.relu(cam).unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_gradcam_examples(
    model: nn.Module,
    manifest_path: Path,
    image_size: int,
    device: torch.device,
    figures_dir: Path,
    n_examples_per_anomaly_class: int = 3,
) -> None:
    manifest = load_split_manifest(manifest_path)
    class_names = sorted(manifest["splits"]["test"].keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    gradcam = GradCAM(model)
    anomaly_classes = [c for c in class_names if c != "normal"]

    for cls_name in anomaly_classes:
        cls_idx = class_to_idx[cls_name]
        test_paths = manifest["splits"]["test"][cls_name]
        saved = 0

        for img_path in test_paths:
            if saved >= n_examples_per_anomaly_class:
                break

            pil_img = Image.open(img_path).convert("RGB")
            image_tensor = tf(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())

            if pred_idx != cls_idx:
                continue

            cam = gradcam.generate(image_tensor, class_idx=pred_idx)

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
            axes[2].set_title(f"Overlay | {cls_name} | conf={confidence:.2f}")

            plt.tight_layout()

            save_path = figures_dir / f"gradcam_{cls_name}_{saved + 1}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Saved Grad-CAM example: %s", save_path)
            saved += 1

        if saved == 0:
            logger.warning("No correctly classified test examples found for Grad-CAM class: %s", cls_name)

    gradcam.remove_hooks()


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]
    manifest_path = results_dir / "cv_split_manifest.json"
    model_path = PROJECT_ROOT / config["cv_anomaly"]["model_save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names, image_size = load_trained_model(model_path, device)

    result = evaluate_on_test_split(
        model=model,
        manifest_path=manifest_path,
        image_size=image_size,
        device=device,
        batch_size=config["cv_anomaly"]["batch_size"],
    )

    (results_dir / "cv_evaluation_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    plot_confusion_matrix(
        cm=result["confusion_matrix"],
        class_names=result["class_names"],
        save_path=figures_dir / "cv_confusion_matrix.png",
    )

    plot_per_class_f1(
        per_class=result["per_class_metrics"],
        save_path=figures_dir / "cv_per_class_f1.png",
    )

    generate_gradcam_examples(
        model=model,
        manifest_path=manifest_path,
        image_size=image_size,
        device=device,
        figures_dir=figures_dir,
        n_examples_per_anomaly_class=3,
    )