"""
Causal Forest implementation using EconML's CausalForestDML.

While Difference-in-Differences gives one average treatment effect, causal
forests estimate heterogeneous treatment effects (HTE), meaning the promotion
effect can vary across stores and items.

This helps answer:
- which stores are most responsive to promotions?
- which items benefit the most?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_causal_forest_data(
    df: pd.DataFrame,
    treatment_col: str,
    target_col: str,
    date_col: str,
    store_col: str,
    item_col: str,
    feature_cols: list[str],
    start_date: str,
    end_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Filter the requested date window and build the arrays needed by EconML.
    """
    sub = df[
        (df[date_col] >= pd.to_datetime(start_date)) &
        (df[date_col] <= pd.to_datetime(end_date))
    ].dropna(subset=[treatment_col, target_col]).reset_index(drop=True)

    available_features = [
        col for col in feature_cols
        if col in sub.columns and col != treatment_col
    ]
    skipped = [col for col in feature_cols if col not in sub.columns]

    if skipped:
        logger.info("Skipped missing causal-forest features: %s", skipped)

    X_df = sub[available_features].fillna(0)
    Y = sub[target_col].values.astype(float)
    T = sub[treatment_col].values.astype(float)
    X = X_df.values.astype(float)
    meta_df = sub[[store_col, item_col]].reset_index(drop=True)

    logger.info(
        "Causal forest input | n=%d | features=%d | treatment_rate=%.3f",
        len(Y),
        X.shape[1],
        T.mean() if len(T) > 0 else 0.0,
    )

    return Y, T, X, meta_df


def run_causal_forest(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    meta_df: pd.DataFrame,
    store_col: str,
    item_col: str,
    n_estimators: int = 500,
    random_state: int = 42,
    max_samples: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a causal forest and aggregate treatment effects to store and item level.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    logger.info(
        "Fitting causal forest | n=%d | features=%d | n_estimators=%d",
        len(Y),
        X.shape[1] if len(X.shape) > 1 else 0,
        n_estimators,
    )

    if len(Y) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(Y), size=max_samples, replace=False)

        Y = Y[idx]
        T = T[idx]
        X = X[idx]
        meta_df = meta_df.iloc[idx].reset_index(drop=True)

        logger.info("Subsampled causal-forest training data to %d rows", max_samples)

    model_y = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=random_state,
    )
    model_t = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=random_state,
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        random_state=random_state,
        inference=True,
        verbose=0,
    )

    cf.fit(Y, T, X=X)
    logger.info("Causal forest fitted successfully")

    te = cf.effect(X)
    te_lower, te_upper = cf.effect_interval(X, alpha=0.05)

    effect_df = meta_df.copy()
    effect_df["promotion_lift_estimate"] = te
    effect_df["te_lower"] = te_lower
    effect_df["te_upper"] = te_upper

    def aggregate_effects(group_col: str) -> pd.DataFrame:
        return (
            effect_df.groupby(group_col)
            .agg(
                promotion_lift_estimate=("promotion_lift_estimate", "mean"),
                te_lower=("te_lower", "mean"),
                te_upper=("te_upper", "mean"),
                n_rows=("promotion_lift_estimate", "count"),
            )
            .reset_index()
            .sort_values("promotion_lift_estimate", ascending=False)
            .reset_index(drop=True)
        )

    store_hte = aggregate_effects(store_col)
    item_hte = aggregate_effects(item_col)

    logger.info(
        "Store HTE range | min=%.4f | max=%.4f",
        store_hte["promotion_lift_estimate"].min() if not store_hte.empty else 0.0,
        store_hte["promotion_lift_estimate"].max() if not store_hte.empty else 0.0,
    )

    logger.info(
        "Item HTE range | min=%.4f | max=%.4f",
        item_hte["promotion_lift_estimate"].min() if not item_hte.empty else 0.0,
        item_hte["promotion_lift_estimate"].max() if not item_hte.empty else 0.0,
    )

    return store_hte, item_hte