from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Causal Demand Forecasting Platform",
    page_icon="📊",
    layout="wide",
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def show_figure(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not yet generated: {path.name}")


st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select page",
    [
        "🏠 Overview",
        "📈 Forecasting",
        "🔬 Causal Inference",
        "💰 Elasticity & Simulation",
        "👁️ Anomaly Detector",
        "🤖 LLM Assistant",
    ],
)

if page == "🏠 Overview":
    st.title("Causal Demand Forecasting & Decision Intelligence Platform")
    st.caption("University of Luxembourg Master's Project")

    st.markdown(
        """
This platform combines:
- demand forecasting
- probabilistic forecasting
- causal inference
- promotion sensitivity / elasticity analysis
- scenario simulation
- visual anomaly detection
- grounded LLM analytics
"""
    )

    w1 = load_json(RESULTS_DIR / "week1_baseline_results.json")
    w2 = load_json(RESULTS_DIR / "week2_forecasting_results.json")
    did = load_json(RESULTS_DIR / "causal_did_result.json")
    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if w1:
            test_row = next((r for r in w1 if r.get("evaluation_split") == "test"), {})
            st.metric("Baseline RMSE", f"{test_row.get('rmse', 'N/A')}")

    with col2:
        if w2:
            lgbm = next((r for r in w2 if r.get("model") == "LightGBM Point"), {})
            if lgbm:
                st.metric("LightGBM RMSE", f"{lgbm.get('rmse', 'N/A')}")

    with col3:
        if did:
            st.metric("DiD Promotion Lift", f"{did.get('estimate', 'N/A')} units/day")

    with col4:
        if cv_eval:
            st.metric("CV Macro F1", f"{cv_eval.get('macro_f1', 'N/A')}")

    st.divider()
    st.subheader("Main Results Table")
    table_path = RESULTS_DIR / "main_results_table.txt"
    if table_path.exists():
        st.code(table_path.read_text(encoding="utf-8"), language=None)
    else:
        st.info("Run scripts/run_llm_pipeline.py to generate the main results table.")

elif page == "📈 Forecasting":
    st.title("📈 Forecasting Results")

    w2 = load_json(RESULTS_DIR / "week2_forecasting_results.json")
    if w2:
        df = pd.DataFrame(w2)
        cols = [c for c in ["model", "rmse", "mae", "mape", "coverage_90", "interval_width", "n_samples"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "lgbm_feature_importance.png", "LightGBM Feature Importance")
    with col2:
        show_figure(FIGURES_DIR / "lgbm_quantile_calibration.png", "Quantile Calibration")

    show_figure(FIGURES_DIR / "sample_quantile_forecast.png", "Sample Quantile Forecast")

elif page == "🔬 Causal Inference":
    st.title("🔬 Causal Inference")

    did = load_json(RESULTS_DIR / "causal_did_result.json")
    placebo = load_json(RESULTS_DIR / "causal_placebo_result.json")
    naive = load_json(RESULTS_DIR / "causal_naive_comparison.json")

    if did and naive:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Naive estimate", f"{naive.get('naive_estimate', 'N/A')} units/day")
        with col2:
            st.metric("DiD estimate", f"{did.get('estimate', 'N/A')} units/day")
        with col3:
            st.metric("p-value", f"{did.get('p_value', 'N/A')}")

    if placebo:
        verdict = placebo.get("verdict", "N/A")
        if placebo.get("passed") is True:
            st.success(f"Placebo test: {verdict}")
        elif placebo.get("passed") is False:
            st.error(f"Placebo test: {verdict}")
        else:
            st.info(f"Placebo test: {verdict}")

    show_figure(FIGURES_DIR / "causal_did_summary.png", "Naive vs DiD vs Placebo")

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "causal_store_hte.png", "Store HTE Ranking")
    with col2:
        show_figure(FIGURES_DIR / "causal_item_hte.png", "Item HTE Ranking")

    item_hte = load_csv(RESULTS_DIR / "causal_item_hte.csv")
    if item_hte is not None:
        st.subheader("Item HTE Table")
        st.dataframe(item_hte.head(20), use_container_width=True)

elif page == "💰 Elasticity & Simulation":
    st.title("💰 Elasticity & Scenario Simulation")

    panel = load_json(RESULTS_DIR / "panel_promotion_sensitivity.json")
    if panel:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Promotion coefficient", f"{panel.get('promotion_coef', 'N/A')}")
        with col2:
            st.metric("Demand uplift", f"{panel.get('pct_demand_change', 'N/A')}%")
        with col3:
            st.metric("Items analysed", f"{panel.get('n_items', 'N/A')}")

    show_figure(FIGURES_DIR / "promotion_sensitivity_by_family.png", "Promotion Sensitivity by Family")

    scenario_df = load_csv(RESULTS_DIR / "week4_scenario_grid.csv")
    if scenario_df is not None:
        st.subheader("Scenario Table")
        st.dataframe(scenario_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "week4_best_simulation_panel.png", "Best Panel Scenario")
    with col2:
        show_figure(FIGURES_DIR / "week4_best_simulation_did.png", "Best DiD Scenario")

elif page == "👁️ Anomaly Detector":
    st.title("👁️ Visual Anomaly Detector")

    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")
    if cv_eval:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{cv_eval.get('accuracy', 'N/A')}")
        with col2:
            st.metric("Macro F1", f"{cv_eval.get('macro_f1', 'N/A')}")
        with col3:
            st.metric("Test Samples", f"{cv_eval.get('n_test_samples', 'N/A')}")

        metrics = cv_eval.get("per_class_metrics") or {}
        if metrics:
            rows = [{"class": cls, **vals} for cls, vals in metrics.items()]
            st.subheader("Per-Class Metrics")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "cv_confusion_matrix.png", "Confusion Matrix")
    with col2:
        show_figure(FIGURES_DIR / "cv_per_class_f1.png", "Per-Class F1")

    st.subheader("Grad-CAM Examples")
    gradcam_files = sorted(FIGURES_DIR.glob("gradcam_*.png"))
    if gradcam_files:
        for img in gradcam_files[:9]:
            show_figure(img, img.name)
    else:
        st.info("No Grad-CAM images found.")

    st.subheader("Flagged Real-Series Anomalies")
    anomaly_results = load_json(RESULTS_DIR / "anomaly_detection_results.json")
    if anomaly_results:
        flagged = [r for r in anomaly_results if r.get("is_anomaly")]
        st.metric("Flagged anomalies", f"{len(flagged)} / {len(anomaly_results)}")
        if flagged:
            st.dataframe(pd.DataFrame(flagged), use_container_width=True)
    else:
        st.info("No anomaly detection results found.")

elif page == "🤖 LLM Assistant":
    st.title("🤖 LLM Analytics Assistant")
    st.caption("Grounded natural-language explanations from saved model outputs")

    try:
        from src.llm.assistant import query_llm
        from src.llm.context_builder import build_context, format_context_for_prompt

        context = build_context(RESULTS_DIR)
        context_str = format_context_for_prompt(context)
        llm_ready = True
    except Exception as exc:
        st.error(f"Failed to load LLM modules: {exc}")
        llm_ready = False

    if llm_ready:
        question = st.text_area(
            "Ask a question about the analytics outputs:",
            value="Generate a weekly executive summary based on all available model outputs.",
            height=100,
        )

        provider = st.selectbox("Provider", ["mock", "anthropic"], index=0)
        model = st.text_input("Model name", value="claude-3-5-haiku-latest")

        if st.button("Ask", type="primary"):
            result = query_llm(
                question=question,
                context_str=context_str,
                provider=provider,
                model=model,
                max_tokens=600,
                temperature=0.2,
            )

            st.subheader("Response")
            st.markdown(result["answer"])

            with st.expander("Model info"):
                st.write(f"Model: {result['model_used']}")
                st.write(f"Input tokens: {result['input_tokens']}")
                st.write(f"Output tokens: {result['output_tokens']}")
                st.write(f"Success: {result['success']}")
                st.write(f"Error: {result['error']}")

        filled_eval = RESULTS_DIR / "llm_human_eval_filled.csv"
        if filled_eval.exists():
            st.divider()
            st.subheader("Human Evaluation Scores")
            eval_df = pd.read_csv(filled_eval)
            score_cols = ["accuracy_1_5", "usefulness_1_5", "groundedness_1_5", "clarity_1_5"]
            available = [c for c in score_cols if c in eval_df.columns]
            if available:
                numeric = eval_df[available].apply(pd.to_numeric, errors="coerce")
                cols = st.columns(len(available))
                labels = ["Accuracy", "Usefulness", "Groundedness", "Clarity"]
                for col, metric_name, score_col in zip(cols, labels, available):
                    with col:
                        st.metric(metric_name, f"{numeric[score_col].mean():.2f} / 5")