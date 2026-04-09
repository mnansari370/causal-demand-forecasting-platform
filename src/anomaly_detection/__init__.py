"""
Visual anomaly detection utilities.
"""

from .generate_synthetic_charts import generate_synthetic_chart_dataset
from .split_dataset import build_split_manifest, load_split_manifest
from .train_detector import train
from .evaluate_detector import (
    load_trained_model,
    evaluate_on_test_split,
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_gradcam_examples,
)
from .inference import run_anomaly_inference

__all__ = [
    "generate_synthetic_chart_dataset",
    "build_split_manifest",
    "load_split_manifest",
    "train",
    "load_trained_model",
    "evaluate_on_test_split",
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "generate_gradcam_examples",
    "run_anomaly_inference",
]