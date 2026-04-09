"""
Core forecasting metrics.

MAPE uses a denominator floor of eps=1.0 to avoid exploding values on
zero-sales rows, which are common in retail demand data.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    eps prevents division by zero for zero-demand rows.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100)


def coverage_at_90(y_true, y_lower, y_upper) -> float:
    """
    Fraction of true values inside the prediction interval.

    For a well-calibrated 90% interval, this should be close to 0.90.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))