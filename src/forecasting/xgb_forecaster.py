from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBPointForecaster:
    """
    XGBoost point forecaster.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 7,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 2000,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        self.model: xgb.XGBRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        self.feature_names_ = list(X_train.columns)
        self.model = xgb.XGBRegressor(**self.params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )

        logger.info("XGBoost point model trained")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained.")
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
        if self.model is None:
            raise ValueError("Model has not been trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved XGBoost model to: %s", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        logger.info("Loaded XGBoost model from: %s", path)