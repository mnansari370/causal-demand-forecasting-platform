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
    LightGBM quantile forecaster with separate models for each quantile.
    Uses 0.05 / 0.50 / 0.95 for better empirical 90% coverage.
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
        self.feature_names_ = list(X_train.columns)

        for q in self.quantiles:
            logger.info("Training LightGBM quantile model for q=%.2f", q)
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
                "Finished q=%.2f | best_iteration=%s",
                q,
                model.best_iteration_,
            )

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.models:
            raise ValueError("Quantile models have not been trained.")

        preds: dict[str, np.ndarray] = {}
        for q in self.quantiles:
            arr = self.models[q].predict(X)
            preds[f"q{q}"] = np.clip(arr, 0, None)
        return preds

    def save(self, dir_path: str | Path) -> None:
        if not self.models:
            raise ValueError("Quantile models have not been trained.")

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for q, model in self.models.items():
            file_name = f"lgbm_q{int(q * 100):02d}.pkl"
            joblib.dump(model, dir_path / file_name)
            logger.info("Saved quantile model: %s", dir_path / file_name)

    def load(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        self.models = {}

        for q in self.quantiles:
            file_name = f"lgbm_q{int(q * 100):02d}.pkl"
            self.models[q] = joblib.load(dir_path / file_name)

        logger.info("Loaded quantile models from: %s", dir_path)