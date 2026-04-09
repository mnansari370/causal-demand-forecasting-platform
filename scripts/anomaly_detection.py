"""
Visual anomaly detection pipeline.

Stages:
1. verify the synthetic dataset exists,
2. build a fixed train/val/test split manifest,
3. train the detector,
4. evaluate on the held-out synthetic test split,
5. generate Grad-CAM examples,
6. run inference on real demand series from the test set.

The synthetic dataset is generated separately by:
    python src/anomaly_detection/generate_synthetic_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.anomaly_detection.evaluate_detector import (
    evaluate_on_test_split,
    generate_gradcam_examples,
    load_trained_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from src.anomaly_detection.inference import run_anomaly_inference
from src.anomaly_detection.split_dataset import build_split_manifest
from src.anomaly_detection.train_detector import train
from src.data.load_data import load_config
from src.utils.logger import get_logger


def count_pngs(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.png")))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    logger = get_logger(
        "anomaly_detection",
        log_dir=PROJECT_ROOT / config["logs"]["log_dir"],
        level=config["logs"]["log_level"],
    )

    anomaly_cfg = config["cv_anomaly"]
    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]
    processed_dir = PROJECT_ROOT / config["data"]["processed_data_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / anomaly_cfg["synthetic_data_dir"]
    model_path = PROJECT_ROOT / anomaly_cfg["model_save_path"]
    manifest_path = results_dir / "cv_split_manifest.json"
    training_history_json = results_dir / "cv_training_history.json"
    training_history_plot = figures_dir / "cv_training_history.png"
    eval_json_path = results_dir / "cv_evaluation_results.json"
    confusion_path = figures_dir / "cv_confusion_matrix.png"
    per_class_plot_path = figures_dir / "cv_per_class_metrics.png"
    anomaly_results_path = results_dir / "anomaly_detection_results.json"
    flagged_dir = figures_dir / "flagged_charts"

    expected_classes = ["normal", "spike", "drop", "structural_break"]

    logger.info("Verifying synthetic anomaly dataset")
    for cls in expected_classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            logger.error("Missing synthetic class directory: %s", cls_dir)
            logger.error("Run: python src/anomaly_detection/generate_synthetic_charts.py")
            sys.exit(1)

        n_images = count_pngs(cls_dir)
        if n_images == 0:
            logger.error("No images found in: %s", cls_dir)
            logger.error("Run: python src/anomaly_detection/generate_synthetic_charts.py")
            sys.exit(1)

        logger.info("Class %-18s images=%d", cls, n_images)

    total_images = sum(count_pngs(data_dir / cls) for cls in expected_classes)
    logger.info("Total synthetic images: %d", total_images)

    logger.info("Building fixed split manifest")
    build_split_manifest(
        data_dir=data_dir,
        output_path=manifest_path,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=config["project"]["random_seed"],
    )

    logger.info("Training anomaly detector")
    train_result = train(
        manifest_path=manifest_path,
        model_save_path=model_path,
        image_size=anomaly_cfg["image_size"],
        batch_size=anomaly_cfg["batch_size"],
        n_epochs=anomaly_cfg["epochs"],
        warmup_epochs=3,
        lr_warmup=1e-3,
        lr_finetune=anomaly_cfg["learning_rate"],
        num_workers=4,
        device_name="auto",
        history_json_path=training_history_json,
        history_plot_path=training_history_plot,
    )

    logger.info(
        "Training summary | best_epoch=%d | best_val_acc=%.4f",
        train_result["best_epoch"],
        train_result["best_val_acc"],
    )

    logger.info("Evaluating on the held-out synthetic test split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_trained_model(model_path, device)

    eval_result = evaluate_on_test_split(
        model=model,
        manifest_path=manifest_path,
        image_size=image_size,
        device=device,
        batch_size=anomaly_cfg["batch_size"],
    )

    eval_json_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        cm=eval_result["confusion_matrix"],
        class_names=eval_result["class_names"],
        save_path=confusion_path,
    )

    plot_per_class_metrics(
        per_class=eval_result["per_class_metrics"],
        save_path=per_class_plot_path,
    )

    print("\n" + "=" * 70)
    print("ANOMALY DETECTOR — SYNTHETIC TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy:   {eval_result['accuracy']:.4f}")
    print(f"Macro F1:   {eval_result['macro_f1']:.4f}")
    print(f"Test size:  {eval_result['n_test_samples']}")
    print("\nPer-class metrics:")
    for cls, metrics in eval_result["per_class_metrics"].items():
        print(
            f"  {cls:<18} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"n={metrics['support']}"
        )
    print("=" * 70)

    logger.info("Generating Grad-CAM examples")
    generate_gradcam_examples(
        model=model,
        manifest_path=manifest_path,
        image_size=image_size,
        device=device,
        figures_dir=figures_dir,
        n_per_class=3,
    )

    logger.info("Running real-series inference on the test set")
    test_path = processed_dir / "test_features.parquet"
    if not test_path.exists():
        logger.warning("Missing %s — skipping real-series inference", test_path)
    else:
        test_df = pd.read_parquet(test_path)

        inference_results = run_anomaly_inference(
            test_df=test_df,
            model=model,
            class_names=class_names,
            image_size=image_size,
            device=device,
            target_col=config["data"]["target_column"],
            store_col=config["data"]["store_column"],
            item_col=config["data"]["item_column"],
            date_col=config["data"]["date_column"],
            top_n=50,
            figures_dir=flagged_dir,
            results_path=anomaly_results_path,
        )

        anomalies = [r for r in inference_results if r["is_anomaly"]]

        print("\n" + "=" * 70)
        print("REAL-SERIES INFERENCE")
        print("=" * 70)
        print(f"Series analysed:      {len(inference_results)}")
        print(f"Anomalies detected:   {len(anomalies)}")

        if anomalies:
            top_anomalies = sorted(
                anomalies,
                key=lambda x: x["confidence"],
                reverse=True,
            )[:5]

            print("\nTop flagged series:")
            for row in top_anomalies:
                print(
                    f"  store={row[config['data']['store_column']]} "
                    f"item={row[config['data']['item_column']]} "
                    f"-> {row['predicted_class']} "
                    f"(conf={row['confidence']:.4f})"
                )
        print("=" * 70)

    logger.info("Anomaly detection pipeline complete")


if __name__ == "__main__":
    main()