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
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    sub = df[
        (df[date_col] >= start_dt) &
        (df[date_col] <= end_dt)
    ].copy()

    sub = sub.dropna(subset=[treatment_col, target_col])

    available_features = [c for c in feature_cols if c in sub.columns]
    missing = [c for c in feature_cols if c not in sub.columns]
    if missing:
        logger.info("Causal forest: feature columns not found (skipped): %s", missing)

    X_df = sub[available_features].fillna(0)

    Y = sub[target_col].values.astype(float)
    T = sub[treatment_col].values.astype(float)
    X = X_df.values.astype(float)

    meta_df = sub[[store_col, item_col]].reset_index(drop=True)

    logger.info(
        "Causal forest data | n=%d | features=%d | treatment_rate=%.3f",
        len(Y), X.shape[1], T.mean(),
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
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    logger.info("Fitting CausalForestDML | n=%d | n_estimators=%d", len(Y), n_estimators)

    if len(Y) > max_samples:
        logger.info("Capping to %d rows for CausalForest (random sample)", max_samples)
        idx = np.random.RandomState(random_state).choice(len(Y), max_samples, replace=False)
        Y_fit, T_fit, X_fit = Y[idx], T[idx], X[idx]
        meta_fit = meta_df.iloc[idx].reset_index(drop=True)
    else:
        Y_fit, T_fit, X_fit = Y, T, X
        meta_fit = meta_df.reset_index(drop=True)

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state),
        n_estimators=n_estimators,
        random_state=random_state,
        inference=True,
        verbose=0,
    )

    cf.fit(Y_fit, T_fit, X=X_fit)
    logger.info("CausalForestDML fitted")

    te_pred = cf.effect(X_fit)
    te_lower, te_upper = cf.effect_interval(X_fit, alpha=0.05)

    meta_fit = meta_fit.copy()
    meta_fit["te"] = te_pred
    meta_fit["te_lower"] = te_lower
    meta_fit["te_upper"] = te_upper

    store_table = (
        meta_fit.groupby(store_col)
        .agg(
            te_mean=("te", "mean"),
            te_lower=("te_lower", "mean"),
            te_upper=("te_upper", "mean"),
            n_rows=("te", "count"),
        )
        .reset_index()
        .sort_values("te_mean", ascending=False)
        .rename(columns={"te_mean": "promotion_lift_estimate"})
        .reset_index(drop=True)
    )

    item_table = (
        meta_fit.groupby(item_col)
        .agg(
            te_mean=("te", "mean"),
            te_lower=("te_lower", "mean"),
            te_upper=("te_upper", "mean"),
            n_rows=("te", "count"),
        )
        .reset_index()
        .sort_values("te_mean", ascending=False)
        .rename(columns={"te_mean": "promotion_lift_estimate"})
        .reset_index(drop=True)
    )

    logger.info(
        "Store HTE | top=%.4f | bottom=%.4f",
        store_table["promotion_lift_estimate"].max(),
        store_table["promotion_lift_estimate"].min(),
    )
    logger.info(
        "Item HTE | top=%.4f | bottom=%.4f",
        item_table["promotion_lift_estimate"].max(),
        item_table["promotion_lift_estimate"].min(),
    )

    return store_table, item_table