"""
LightGBM quantile forecaster.

This module trains separate models for lower, median, and upper quantiles.
Together they form a prediction interval that can be used to quantify
forecast uncertainty.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LGBMQuantileForecaster:
    """
    LightGBM quantile forecaster with one model per quantile.
    """

    def __init__(self, quantiles: list[float] | None = None) -> None:
        self.quantiles = quantiles or [0.05, 0.5, 0.95]
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: list[str] = []

        self.base_params = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_estimators": 2000,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Train one LightGBM model per target quantile.
        """
        self.feature_names_ = list(X_train.columns)

        for q in self.quantiles:
            logger.info("Training quantile model q=%.2f", q)

            params = {**self.base_params, "alpha": q}
            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )

            self.models[q] = model

            logger.info(
                "Finished quantile q=%.2f | best_iteration=%s",
                q,
                model.best_iteration_,
            )

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict all configured quantiles.
        """
        if not self.models:
            raise RuntimeError("Quantile models have not been trained yet.")

        preds: dict[str, np.ndarray] = {}
        for q in self.quantiles:
            preds[f"q{q}"] = np.clip(self.models[q].predict(X), 0, None)

        return preds

    def save(self, dir_path: str | Path) -> None:
        """
        Save all quantile models into a directory.
        """
        if not self.models:
            raise RuntimeError("Quantile models have not been trained yet.")

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for q, model in self.models.items():
            file_name = f"lgbm_q{int(q * 100):02d}.pkl"
            joblib.dump(model, dir_path / file_name)
            logger.info("Saved quantile model: %s", dir_path / file_name)

    def load(self, dir_path: str | Path) -> None:
        """
        Load all configured quantile models from a directory.
        """
        dir_path = Path(dir_path)
        self.models = {}

        for q in self.quantiles:
            file_name = f"lgbm_q{int(q * 100):02d}.pkl"
            self.models[q] = joblib.load(dir_path / file_name)

        logger.info("Loaded quantile models from: %s", dir_path)