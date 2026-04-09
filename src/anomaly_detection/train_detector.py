"""
Train a visual anomaly detector on synthetic chart images.

We use a pretrained ResNet-18 and fine-tune it in two phases:

1. Warmup phase:
   Freeze the backbone and train only the classifier head.
2. Fine-tuning phase:
   Unfreeze the full network and continue training with a smaller learning rate.

This is more stable than training the full network immediately.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.anomaly_detection.split_dataset import load_split_manifest
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SyntheticChartDataset(Dataset):
    """
    Dataset backed by the split manifest.
    """

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


def get_transforms(image_size: int = 224) -> dict[str, transforms.Compose]:
    """
    Use only mild augmentation.

    We avoid horizontal flips because that would reverse time direction
    in a time-series chart.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {"train": train_tf, "eval": eval_tf}


def build_dataloaders_from_manifest(
    manifest_path: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and validation dataloaders from the saved manifest.
    """
    manifest = load_split_manifest(manifest_path)
    transforms_map = get_transforms(image_size=image_size)

    class_names = sorted(manifest["splits"]["train"].keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []

    for cls in class_names:
        train_samples.extend(
            [(path, class_to_idx[cls]) for path in manifest["splits"]["train"][cls]]
        )
        val_samples.extend(
            [(path, class_to_idx[cls]) for path in manifest["splits"]["val"][cls]]
        )

    train_ds = SyntheticChartDataset(train_samples, transform=transforms_map["train"])
    val_ds = SyntheticChartDataset(val_samples, transform=transforms_map["eval"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Dataloaders ready | train=%d | val=%d | classes=%s",
        len(train_ds),
        len(val_ds),
        class_names,
    )

    return train_loader, val_loader, class_names


def build_resnet18(n_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """
    Build a ResNet-18 with a custom classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "ResNet-18 built | trainable=%d / %d | freeze_backbone=%s",
        n_trainable,
        n_total,
        freeze_backbone,
    )
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """
    Run one training or validation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train and optimizer is not None:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(
    manifest_path: str | Path,
    model_save_path: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    n_epochs: int = 10,
    warmup_epochs: int = 3,
    lr_warmup: float = 1e-3,
    lr_finetune: float = 1e-4,
    num_workers: int = 4,
    device_name: str = "auto",
    history_json_path: str | Path | None = None,
    history_plot_path: str | Path | None = None,
) -> dict:
    """
    Train the anomaly detector and save the best checkpoint.
    """
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    pin_memory = device.type == "cuda"

    logger.info("Training device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    train_loader, val_loader, class_names = build_dataloaders_from_manifest(
        manifest_path=manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_resnet18(n_classes=len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_warmup,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(warmup_epochs, 1),
        eta_min=1e-6,
    )

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, n_epochs + 1):
        if epoch == warmup_epochs + 1:
            logger.info("Unfreezing backbone for fine-tuning")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(n_epochs - warmup_epochs, 1),
                eta_min=1e-6,
            )

        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step()

        phase = "WARMUP" if epoch <= warmup_epochs else "FINETUNE"

        logger.info(
            "Epoch %2d/%d [%s] | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            n_epochs,
            phase,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            model_save_path = Path(model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "image_size": image_size,
                    "architecture": "resnet18",
                },
                model_save_path,
            )

            logger.info("Saved best model | epoch=%d | val_acc=%.4f", epoch, val_acc)

    result = {
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "history": history,
    }

    if history_json_path is not None:
        history_json_path = Path(history_json_path)
        history_json_path.parent.mkdir(parents=True, exist_ok=True)
        history_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if history_plot_path is not None:
        _plot_history(history, history_plot_path)

    logger.info(
        "Training complete | best_epoch=%d | best_val_acc=%.4f",
        best_epoch,
        best_val_acc,
    )

    return result


def _plot_history(history: list[dict], save_path: str | Path) -> None:
    """
    Save a simple training-history plot.
    """
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Val Loss")
    ax.plot(epochs, [row["train_acc"] for row in history], label="Train Acc")
    ax.plot(epochs, [row["val_acc"] for row in history], label="Val Acc")

    ax.set_xlabel("Epoch")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved training history plot: %s", save_path)