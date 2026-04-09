"""
Train/validation/test split builder for synthetic anomaly charts.

We store the split as a manifest JSON so the exact dataset split remains
fixed and reproducible across runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
random.seed(SEED)


def _list_pngs(class_dir: Path) -> list[str]:
    return sorted(str(p.resolve()) for p in class_dir.glob("*.png"))


def build_split_manifest(
    data_dir: str | Path,
    output_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Build a fixed split manifest from class folders.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {data_dir}")

    rng = random.Random(seed)

    manifest = {
        "seed": seed,
        "data_dir": str(data_dir.resolve()),
        "fractions": {
            "train": train_frac,
            "val": val_frac,
            "test": test_frac,
        },
        "splits": {
            "train": {},
            "val": {},
            "test": {},
        },
    }

    logger.info("Building split manifest from: %s", data_dir)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = _list_pngs(class_dir)

        if not files:
            raise ValueError(f"No PNG files found in class directory: {class_dir}")

        rng.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        n_test = n_total - n_train - n_val

        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError(
                f"Split too small for class '{class_name}': "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        manifest["splits"]["train"][class_name] = files[:n_train]
        manifest["splits"]["val"][class_name] = files[n_train:n_train + n_val]
        manifest["splits"]["test"][class_name] = files[n_train + n_val:]

        logger.info(
            "Class %-18s total=%4d | train=%4d | val=%4d | test=%4d",
            class_name,
            n_total,
            n_train,
            n_val,
            n_test,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved split manifest: %s", output_path)
    return manifest


def load_split_manifest(manifest_path: str | Path) -> dict:
    """
    Load an existing split manifest.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))