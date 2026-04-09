from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
random.seed(SEED)


def _list_pngs(class_dir: Path) -> list[str]:
    return sorted(str(p.resolve()) for p in class_dir.glob("*.png"))


def build_split_manifest(
    data_dir: Path,
    output_path: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Build a fixed stratified split manifest for synthetic anomaly images.

    Expected structure:
      data_dir/
        normal/
        spike/
        drop/
        structural_break/

    Output JSON contains absolute file paths grouped by split and class.
    """

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {data_dir}")

    rng = random.Random(seed)

    manifest: dict = {
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

    logger.info("Building CV split manifest from: %s", data_dir)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = _list_pngs(class_dir)

        if len(files) == 0:
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

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        manifest["splits"]["train"][class_name] = train_files
        manifest["splits"]["val"][class_name] = val_files
        manifest["splits"]["test"][class_name] = test_files

        logger.info(
            "Class %-18s total=%4d | train=%4d | val=%4d | test=%4d",
            class_name, n_total, n_train, n_val, n_test
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved CV split manifest: %s", output_path)

    return manifest


def load_split_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from src.data.load_data import load_config

    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    data_dir = PROJECT_ROOT / config["cv_anomaly"]["synthetic_data_dir"]
    output_path = PROJECT_ROOT / config["evaluation"]["results_dir"] / "cv_split_manifest.json"

    build_split_manifest(
        data_dir=data_dir,
        output_path=output_path,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=config["project"]["random_seed"],
    )