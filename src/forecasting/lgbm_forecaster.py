"""
LightGBM point forecaster.

LightGBM is the main forecasting model in this project because it handles
mixed tabular features well, trains quickly, and gives strong performance
on retail demand data with lag, rolling, calendar, and promotion features.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LGBMPointForecaster:
    """
    LightGBM regressor for point forecasting.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_estimators": 2000,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Train the model with early stopping on the validation set.
        """
        self.feature_names_ = list(X_train.columns)
        self.model = lgb.LGBMRegressor(**self.params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        logger.info(
            "LightGBM trained | best_iteration=%s | val_rmse=%.4f",
            self.model.best_iteration_,
            self.model.best_score_["valid_0"]["rmse"],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict non-negative unit sales.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return np.clip(self.model.predict(X), 0, None)

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances as a sorted dataframe.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

        logger.info("Saved LightGBM model: %s", path)

    def load(self, path: str | Path) -> None:
        """
        Load a previously saved model.
        """
        self.model = joblib.load(path)
        logger.info("Loaded LightGBM model: %s", path)