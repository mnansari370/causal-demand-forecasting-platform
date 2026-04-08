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
    Prepare Y, T, X matrices for CausalForestDML.
    """

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    sub = df[
        (df[date_col] >= start_dt) &
        (df[date_col] <= end_dt)
    ].copy()

    sub = sub.dropna(subset=[treatment_col, target_col]).reset_index(drop=True)

    # Remove treatment column from features
    available = [c for c in feature_cols if c in sub.columns and c != treatment_col]
    skipped = [c for c in feature_cols if c not in sub.columns]

    if skipped:
        logger.info("Skipped feature columns (not found): %s", skipped)

    X_df = sub[available].fillna(0)

    Y = sub[target_col].values.astype(float)
    T = sub[treatment_col].values.astype(float)
    X = X_df.values.astype(float)

    meta_df = sub[[store_col, item_col]].reset_index(drop=True)

    logger.info(
        "Causal forest data — n=%d | p=%d | treatment_rate=%.3f",
        len(Y), X.shape[1], T.mean()
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
    Fit CausalForestDML and return store-level and item-level HTE tables.
    """

    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    logger.info(
        "Fitting CausalForest | n=%d | estimators=%d",
        len(Y), n_estimators
    )

    # Subsample to avoid memory issues
    if len(Y) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(Y), max_samples, replace=False)

        Y_fit = Y[idx]
        T_fit = T[idx]
        X_fit = X[idx]
        meta_fit = meta_df.iloc[idx].reset_index(drop=True)

        logger.info("Subsampled to %d rows", max_samples)
    else:
        Y_fit, T_fit, X_fit = Y, T, X
        meta_fit = meta_df.copy()

    # First-stage models
    model_y = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
    )

    model_t = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
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

    cf.fit(Y_fit, T_fit, X=X_fit)
    logger.info("Causal forest fitted")

    # Individual treatment effects
    te = cf.effect(X_fit)
    te_lower, te_upper = cf.effect_interval(X_fit, alpha=0.05)

    meta_fit = meta_fit.copy()
    meta_fit["te"] = te
    meta_fit["te_lower"] = te_lower
    meta_fit["te_upper"] = te_upper

    # --- Store-level aggregation ---
    store_hte = (
        meta_fit.groupby(store_col)
        .agg(
            promotion_lift_estimate=("te", "mean"),
            te_lower=("te_lower", "mean"),
            te_upper=("te_upper", "mean"),
            n_rows=("te", "count"),
        )
        .reset_index()
        .sort_values("promotion_lift_estimate", ascending=False)
        .reset_index(drop=True)
    )

    # --- Item-level aggregation ---
    item_hte = (
        meta_fit.groupby(item_col)
        .agg(
            promotion_lift_estimate=("te", "mean"),
            te_lower=("te_lower", "mean"),
            te_upper=("te_upper", "mean"),
            n_rows=("te", "count"),
        )
        .reset_index()
        .sort_values("promotion_lift_estimate", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Store HTE range: %.4f → %.4f",
        store_hte["promotion_lift_estimate"].min(),
        store_hte["promotion_lift_estimate"].max(),
    )

    logger.info(
        "Item HTE range: %.4f → %.4f",
        item_hte["promotion_lift_estimate"].min(),
        item_hte["promotion_lift_estimate"].max(),
    )

    return store_hte, item_hte