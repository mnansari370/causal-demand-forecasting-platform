"""
Build a structured context for the LLM from saved result files.

This keeps the LLM grounded: it does not inspect raw data or run new models.
It only explains the outputs already produced by the pipeline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_json_safe(path: str | Path) -> dict | list | None:
    path = Path(path)

    if not path.exists():
        logger.info("JSON not found, skipping: %s", path)
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
        return None


def load_csv_safe(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)

    if not path.exists():
        logger.info("CSV not found, skipping: %s", path)
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load CSV %s: %s", path, exc)
        return None


def _round_or_none(x: Any, ndigits: int = 4) -> float | None:
    if x is None:
        return None

    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, ndigits)
    except Exception:
        return None


def _exp_pct_from_coef(coef: Any) -> float | None:
    coef_f = _round_or_none(coef, 8)
    if coef_f is None:
        return None
    return round((math.exp(coef_f) - 1.0) * 100.0, 2)


def _top_records(df: pd.DataFrame, n: int = 5) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.head(n).to_dict(orient="records")


def _bottom_records(df: pd.DataFrame, n: int = 5) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.tail(n).to_dict(orient="records")


def build_context(results_dir: str | Path) -> dict:
    """
    Collect saved outputs into one structured context dictionary.
    """
    results_dir = Path(results_dir)
    context: dict[str, Any] = {}

    forecasting = load_json_safe(results_dir / "forecasting_results.json")
    if isinstance(forecasting, list) and forecasting:
        models = []
        for row in forecasting:
            models.append(
                {
                    "model": row.get("model"),
                    "rmse": _round_or_none(row.get("rmse")),
                    "mae": _round_or_none(row.get("mae")),
                    "mape": _round_or_none(row.get("mape"), 2),
                    "coverage_90": _round_or_none(row.get("coverage_90")),
                    "interval_width": _round_or_none(row.get("interval_width")),
                    "n_samples": row.get("n_samples"),
                }
            )

        context["forecasting_models"] = models

        baseline = next((r for r in models if "Naive" in str(r.get("model", ""))), None)
        lgbm_point = next((r for r in models if r.get("model") == "LightGBM Point"), None)
        lgbm_quant = next((r for r in models if "Quantile" in str(r.get("model", ""))), None)

        if baseline:
            context["baseline_forecasting"] = {
                "model": baseline.get("model"),
                "test_rmse": baseline.get("rmse"),
                "test_mae": baseline.get("mae"),
                "test_mape": baseline.get("mape"),
            }

        if baseline and lgbm_point:
            improvement = None
            if baseline.get("rmse") and lgbm_point.get("rmse"):
                improvement = round((1 - lgbm_point["rmse"] / baseline["rmse"]) * 100, 1)

            context["best_forecasting_model"] = {
                "model": lgbm_point.get("model"),
                "rmse": lgbm_point.get("rmse"),
                "mae": lgbm_point.get("mae"),
                "improvement_over_baseline_pct": improvement,
            }

        if lgbm_quant:
            cov = lgbm_quant.get("coverage_90")
            context["probabilistic_forecasting"] = {
                "model": lgbm_quant.get("model"),
                "coverage_90": cov,
                "interval_width": lgbm_quant.get("interval_width"),
                "interpretation": (
                    f"{cov * 100:.1f}% of actual values fell inside the nominal 90% interval."
                    if cov is not None
                    else "Coverage information is not available."
                ),
            }

    did = load_json_safe(results_dir / "causal_did_result.json")
    if isinstance(did, dict) and did:
        estimate = _round_or_none(did.get("estimate"))
        ci_low = _round_or_none(did.get("ci_low"))
        ci_high = _round_or_none(did.get("ci_high"))
        p_value = _round_or_none(did.get("p_value"), 4)

        context["causal_did"] = {
            "method": "Difference-in-Differences (OLS regression)",
            "att_estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "significant": did.get("significant"),
            "n_obs": did.get("n_obs"),
            "interpretation": (
                f"Promotions increased demand by {estimate:+.3f} units/day "
                f"(95% CI [{ci_low:+.3f}, {ci_high:+.3f}], p={p_value:.4f})."
                if None not in (estimate, ci_low, ci_high, p_value)
                else "DiD results are available but incomplete."
            ),
        }

    placebo = load_json_safe(results_dir / "causal_placebo_result.json")
    if isinstance(placebo, dict) and placebo:
        passed = placebo.get("passed")
        context["causal_placebo_test"] = {
            "verdict": placebo.get("verdict"),
            "placebo_estimate": _round_or_none(placebo.get("estimate")),
            "p_value": _round_or_none(placebo.get("p_value"), 4),
            "passed": passed,
            "interpretation": (
                "Placebo test passed, which supports the credibility of the DiD design."
                if passed is True
                else "Placebo test did not pass cleanly, so the DiD estimate should be interpreted cautiously."
                if passed is False
                else "Placebo result is incomplete."
            ),
        }

    naive = load_json_safe(results_dir / "causal_naive_comparison.json")
    if isinstance(naive, dict) and naive and "causal_did" in context:
        naive_est = _round_or_none(naive.get("naive_estimate"))
        did_est = context["causal_did"].get("att_estimate")

        bias_pct = None
        if naive_est not in (None, 0) and did_est is not None:
            bias_pct = round(abs((naive_est - did_est) / naive_est) * 100, 1)

        context["selection_bias"] = {
            "naive_estimate": naive_est,
            "causal_estimate": did_est,
            "bias_pct": bias_pct,
            "interpretation": (
                f"The naive estimate overstated the effect by about {bias_pct}%."
                if bias_pct is not None
                else "Selection-bias comparison is available but incomplete."
            ),
        }

    store_hte = load_csv_safe(results_dir / "causal_store_hte.csv")
    if store_hte is not None and not store_hte.empty:
        context["store_promotion_sensitivity"] = {
            "n_stores": int(len(store_hte)),
            "top_stores": _top_records(store_hte, n=5),
            "interpretation": "Stores ranked by heterogeneous treatment effect of promotions.",
        }

    item_hte = load_csv_safe(results_dir / "causal_item_hte.csv")
    if item_hte is not None and not item_hte.empty:
        context["item_promotion_sensitivity"] = {
            "n_items": int(len(item_hte)),
            "top_10": _top_records(item_hte, n=10),
            "bottom_10": _bottom_records(item_hte, n=10),
            "interpretation": "Items ranked by heterogeneous treatment effect of promotions.",
        }

    panel = load_json_safe(results_dir / "panel_promotion_sensitivity.json")
    if isinstance(panel, dict) and panel:
        context["panel_promotion_sensitivity"] = {
            "method": "Panel OLS with item fixed effects",
            "promotion_coef": _round_or_none(panel.get("promotion_coef")),
            "pct_demand_change": _round_or_none(panel.get("pct_demand_change"), 2),
            "ci_low_pct": _exp_pct_from_coef(panel.get("ci_low")),
            "ci_high_pct": _exp_pct_from_coef(panel.get("ci_high")),
            "p_value": _round_or_none(panel.get("p_value"), 4),
            "n_items": panel.get("n_items"),
            "significant": panel.get("significant"),
            "interpretation": (
                f"Promotions increased demand by about {_round_or_none(panel.get('pct_demand_change'), 2):+.2f}% on average."
                if panel.get("pct_demand_change") is not None
                else "Panel promotion sensitivity results are incomplete."
            ),
        }

    family_df = load_csv_safe(results_dir / "promotion_sensitivity_by_family.csv")
    if family_df is not None and not family_df.empty:
        sig_df = family_df[family_df["significant"] == True].copy() if "significant" in family_df.columns else family_df.copy()

        if "pct_demand_change" in sig_df.columns:
            sig_df = sig_df.sort_values("pct_demand_change", ascending=False).reset_index(drop=True)

        context["family_promotion_sensitivity"] = {
            "n_families": int(len(family_df)),
            "n_significant": int(sig_df["significant"].sum()) if "significant" in sig_df.columns else None,
            "top_5_families": _top_records(sig_df, n=5),
            "bottom_5_families": _bottom_records(sig_df, n=5),
        }

    cv_eval = load_json_safe(results_dir / "cv_evaluation_results.json")
    if isinstance(cv_eval, dict) and cv_eval:
        per_class = cv_eval.get("per_class_metrics") or {}
        context["anomaly_detector"] = {
            "architecture": "ResNet-18 fine-tuned on synthetic anomaly chart images",
            "accuracy": _round_or_none(cv_eval.get("accuracy")),
            "macro_f1": _round_or_none(cv_eval.get("macro_f1")),
            "per_class_f1": {
                cls: _round_or_none(metrics.get("f1"))
                for cls, metrics in per_class.items()
            },
            "n_test_samples": cv_eval.get("n_test_samples"),
            "classes": cv_eval.get("class_names"),
        }

    anomaly_results = load_json_safe(results_dir / "anomaly_detection_results.json")
    if isinstance(anomaly_results, list) and anomaly_results:
        flagged = [r for r in anomaly_results if r.get("is_anomaly")]
        normal = [r for r in anomaly_results if not r.get("is_anomaly")]

        flagged_sorted = sorted(flagged, key=lambda x: x.get("confidence", 0), reverse=True)

        context["detected_anomalies"] = {
            "total_series_analysed": int(len(anomaly_results)),
            "n_anomalies": int(len(flagged)),
            "n_normal": int(len(normal)),
            "anomaly_rate_pct": round(len(flagged) / len(anomaly_results) * 100, 1),
            "flagged_series": flagged_sorted[:10],
        }

    logger.info("Context built | keys=%s", list(context.keys()))
    return context


def format_context_for_prompt(context: dict) -> str:
    """
    Convert the structured context into a compact prompt string.
    """
    lines: list[str] = []
    lines.append("=== RETAIL ANALYTICS CONTEXT ===\n\n")

    if "baseline_forecasting" in context:
        b = context["baseline_forecasting"]
        lines.append("FORECASTING BASELINE:\n")
        lines.append(
            f"- Model: {b['model']}\n"
            f"- Test RMSE: {b['test_rmse']}\n"
            f"- Test MAE: {b['test_mae']}\n"
            f"- Test MAPE: {b['test_mape']}%\n\n"
        )

    if "best_forecasting_model" in context:
        m = context["best_forecasting_model"]
        lines.append("BEST FORECASTING MODEL:\n")
        lines.append(
            f"- Model: {m['model']}\n"
            f"- RMSE: {m['rmse']}\n"
            f"- MAE: {m['mae']}\n"
            f"- Improvement over baseline: {m['improvement_over_baseline_pct']}%\n\n"
        )

    if "probabilistic_forecasting" in context:
        p = context["probabilistic_forecasting"]
        lines.append("PROBABILISTIC FORECASTING:\n")
        lines.append(
            f"- Model: {p['model']}\n"
            f"- Coverage@90: {p['coverage_90']}\n"
            f"- Interval width: {p['interval_width']}\n"
            f"- Interpretation: {p['interpretation']}\n\n"
        )

    if "causal_did" in context:
        d = context["causal_did"]
        lines.append("CAUSAL INFERENCE (DiD):\n")
        lines.append(
            f"- Method: {d['method']}\n"
            f"- ATT estimate: {d['att_estimate']} units/day\n"
            f"- 95% CI: [{d['ci_low']}, {d['ci_high']}]\n"
            f"- p-value: {d['p_value']}\n"
            f"- Significant: {d['significant']}\n"
            f"- Interpretation: {d['interpretation']}\n\n"
        )

    if "causal_placebo_test" in context:
        pt = context["causal_placebo_test"]
        lines.append("PLACEBO TEST:\n")
        lines.append(
            f"- Verdict: {pt['verdict']}\n"
            f"- Placebo estimate: {pt['placebo_estimate']}\n"
            f"- p-value: {pt['p_value']}\n"
            f"- Interpretation: {pt['interpretation']}\n\n"
        )

    if "selection_bias" in context:
        sb = context["selection_bias"]
        lines.append("SELECTION BIAS:\n")
        lines.append(
            f"- Naive estimate: {sb['naive_estimate']} units/day\n"
            f"- Causal estimate: {sb['causal_estimate']} units/day\n"
            f"- Bias percent: {sb['bias_pct']}%\n"
            f"- Interpretation: {sb['interpretation']}\n\n"
        )

    if "panel_promotion_sensitivity" in context:
        pe = context["panel_promotion_sensitivity"]
        lines.append("PANEL PROMOTION SENSITIVITY:\n")
        lines.append(
            f"- Method: {pe['method']}\n"
            f"- Promotion coefficient: {pe['promotion_coef']}\n"
            f"- Average demand change: {pe['pct_demand_change']}%\n"
            f"- Approx 95% CI in percent: [{pe['ci_low_pct']}%, {pe['ci_high_pct']}%]\n"
            f"- p-value: {pe['p_value']}\n"
            f"- Interpretation: {pe['interpretation']}\n\n"
        )

    if "family_promotion_sensitivity" in context:
        fe = context["family_promotion_sensitivity"]
        lines.append("FAMILY-LEVEL PROMOTION SENSITIVITY:\n")
        lines.append(
            f"- Families estimated: {fe['n_families']}\n"
            f"- Significant families: {fe['n_significant']}\n"
        )
        if fe.get("top_5_families"):
            lines.append("- Top 5 families:\n")
            for row in fe["top_5_families"]:
                lines.append(
                    f"  - {row.get('family', 'N/A')}: {row.get('pct_demand_change', 'N/A')}% demand change\n"
                )
        lines.append("\n")

    if "store_promotion_sensitivity" in context:
        sh = context["store_promotion_sensitivity"]
        lines.append("STORE-LEVEL HTE:\n")
        lines.append(f"- Stores analysed: {sh['n_stores']}\n")
        for row in sh.get("top_stores", []):
            lines.append(
                f"  - Store {row.get('store_nbr', '?')}: {row.get('promotion_lift_estimate', 'N/A')} units/day\n"
            )
        lines.append("\n")

    if "item_promotion_sensitivity" in context:
        ih = context["item_promotion_sensitivity"]
        lines.append("ITEM-LEVEL HTE:\n")
        lines.append(f"- Items analysed: {ih['n_items']}\n")
        if ih.get("top_10"):
            lines.append("- Top 5 items:\n")
            for row in ih["top_10"][:5]:
                lines.append(
                    f"  - Item {row.get('item_nbr', '?')}: {row.get('promotion_lift_estimate', 'N/A')} units/day\n"
                )
        lines.append("\n")

    if "anomaly_detector" in context:
        ad = context["anomaly_detector"]
        lines.append("ANOMALY DETECTOR:\n")
        lines.append(
            f"- Architecture: {ad['architecture']}\n"
            f"- Accuracy: {ad['accuracy']}\n"
            f"- Macro F1: {ad['macro_f1']}\n"
            f"- Per-class F1: {ad['per_class_f1']}\n\n"
        )

    if "detected_anomalies" in context:
        da = context["detected_anomalies"]
        lines.append("DETECTED ANOMALIES ON REAL TEST DATA:\n")
        lines.append(
            f"- Series analysed: {da['total_series_analysed']}\n"
            f"- Anomalies detected: {da['n_anomalies']}\n"
            f"- Anomaly rate: {da['anomaly_rate_pct']}%\n"
        )
        if da.get("flagged_series"):
            lines.append("- Top flagged series:\n")
            for row in da["flagged_series"][:5]:
                lines.append(
                    f"  - Store {row.get('store_nbr', '?')} Item {row.get('item_nbr', '?')}: "
                    f"{row.get('predicted_class', '?')} (confidence={row.get('confidence', 'N/A')})\n"
                )
        lines.append("\n")

    lines.append("=== END CONTEXT ===")
    return "".join(lines)