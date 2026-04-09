"""
Streamlit demo interface for the demand forecasting platform.

Run with:
    streamlit run app/streamlit_app.py

The app reads from precomputed outputs in outputs/evaluation/ and
outputs/figures/. All heavy computation happens in the scripts, not
inside the app itself. This keeps the interface fast and reproducible.

Pages:
- Overview
- Forecasting
- Causal Inference
- Promotion Sensitivity
- Anomaly Detector
- LLM Assistant
"""
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
        st.info(f"Figure not found: {path.name}. Run the corresponding pipeline first.")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Forecasting",
        "🔬 Causal Inference",
        "💰 Promotion Sensitivity",
        "👁️ Anomaly Detector",
        "🤖 LLM Assistant",
    ],
)

if page == "🏠 Overview":
    st.title("Causal Demand Forecasting & Decision Intelligence Platform")
    st.caption("University of Luxembourg — MSc Computer Science Project")

    st.markdown(
        "This platform combines **demand forecasting**, **causal inference**, "
        "**promotion sensitivity analysis**, **visual anomaly detection**, and "
        "**an LLM-based analytics assistant** on the Favorita retail dataset."
    )

    forecasting = load_json(RESULTS_DIR / "forecasting_results.json")
    did = load_json(RESULTS_DIR / "causal_did_result.json")
    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")

    col1, col2, col3, col4 = st.columns(4)

    if forecasting:
        naive = next((row for row in forecasting if "Naive" in str(row.get("model", ""))), {})
        lgbm = next((row for row in forecasting if row.get("model") == "LightGBM Point"), {})

        with col1:
            st.metric("Baseline RMSE", f"{naive.get('rmse', 'N/A')}")
        with col2:
            st.metric("LightGBM RMSE", f"{lgbm.get('rmse', 'N/A')}")

    if did:
        with col3:
            st.metric("DiD Promotion Lift", f"{did.get('estimate', 'N/A')} units/day")

    if cv_eval:
        with col4:
            st.metric("CV Macro F1", f"{cv_eval.get('macro_f1', 'N/A')}")

    st.divider()

    table_path = RESULTS_DIR / "main_results_table.txt"
    if table_path.exists():
        st.subheader("Main Results Table")
        st.code(table_path.read_text(encoding="utf-8"), language=None)
    else:
        st.info("Run scripts/llm_pipeline.py to generate the main results table.")

elif page == "📈 Forecasting":
    st.title("Forecasting Results")

    forecasting = load_json(RESULTS_DIR / "forecasting_results.json")
    if forecasting:
        df = pd.DataFrame(forecasting)
        cols = [
            c for c in
            ["model", "rmse", "mae", "mape", "coverage_90", "interval_width", "n_samples"]
            if c in df.columns
        ]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No forecasting results found. Run scripts/forecasting.py.")

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "lgbm_feature_importance.png", "LightGBM Feature Importance")
    with col2:
        show_figure(FIGURES_DIR / "lgbm_quantile_calibration.png", "Quantile Calibration")

    show_figure(
        FIGURES_DIR / "sample_quantile_forecast.png",
        "Sample Forecast with 90% Prediction Interval",
    )

elif page == "🔬 Causal Inference":
    st.title("Causal Inference")

    did = load_json(RESULTS_DIR / "causal_did_result.json")
    placebo = load_json(RESULTS_DIR / "causal_placebo_result.json")
    naive = load_json(RESULTS_DIR / "causal_naive_comparison.json")

    if did and naive:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Naive estimate", f"{naive.get('naive_estimate', 'N/A')} u/day")
        with col2:
            st.metric("DiD ATT", f"{did.get('estimate', 'N/A')} u/day")
        with col3:
            st.metric("p-value", f"{did.get('p_value', 'N/A')}")
        with col4:
            if naive.get("naive_estimate") and did.get("estimate"):
                bias = abs((naive["naive_estimate"] - did["estimate"]) / naive["naive_estimate"] * 100)
                st.metric("Selection bias", f"{bias:.1f}%")

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
        show_figure(FIGURES_DIR / "causal_store_hte.png", "Store Promotion Sensitivity")
    with col2:
        show_figure(FIGURES_DIR / "causal_item_hte.png", "Item Promotion Sensitivity")

    item_hte = load_csv(RESULTS_DIR / "causal_item_hte.csv")
    if item_hte is not None:
        st.subheader("Item HTE Table")
        st.dataframe(item_hte.head(20), use_container_width=True)

elif page == "💰 Promotion Sensitivity":
    st.title("Promotion Sensitivity & Scenario Simulation")

    panel = load_json(RESULTS_DIR / "panel_promotion_sensitivity.json")
    if panel:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Promotion coefficient", f"{panel.get('promotion_coef', 'N/A')}")
        with col2:
            st.metric("Demand uplift", f"{panel.get('pct_demand_change', 'N/A')}%")
        with col3:
            st.metric("Items analysed", f"{panel.get('n_items', 'N/A')}")

    show_figure(
        FIGURES_DIR / "promotion_sensitivity_by_family.png",
        "Promotion Sensitivity by Family",
    )

    scenario_df = load_csv(RESULTS_DIR / "scenario_grid.csv")
    if scenario_df is not None:
        st.subheader("Scenario Grid")
        st.dataframe(scenario_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "simulation_panel.png", "Panel-Based Promotion Scenario")
    with col2:
        show_figure(FIGURES_DIR / "simulation_did.png", "DiD-Based Promotion Scenario")

elif page == "👁️ Anomaly Detector":
    st.title("Visual Anomaly Detector")

    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")
    if cv_eval:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{cv_eval.get('accuracy', 'N/A')}")
        with col2:
            st.metric("Macro F1", f"{cv_eval.get('macro_f1', 'N/A')}")
        with col3:
            st.metric("Test samples", f"{cv_eval.get('n_test_samples', 'N/A')}")

        per_class = cv_eval.get("per_class_metrics") or {}
        if per_class:
            st.subheader("Per-class Metrics")
            st.dataframe(
                pd.DataFrame([{"class": cls, **metrics} for cls, metrics in per_class.items()]),
                use_container_width=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        show_figure(FIGURES_DIR / "cv_confusion_matrix.png", "Confusion Matrix")
    with col2:
        show_figure(FIGURES_DIR / "cv_per_class_metrics.png", "Per-class Metrics")

    st.subheader("Grad-CAM Examples")
    gradcam_files = sorted(FIGURES_DIR.glob("gradcam_*.png"))
    if gradcam_files:
        cols = st.columns(3)
        for i, img in enumerate(gradcam_files[:9]):
            with cols[i % 3]:
                show_figure(img, img.stem)
    else:
        st.info("No Grad-CAM images found.")

    anomaly_results = load_json(RESULTS_DIR / "anomaly_detection_results.json")
    if anomaly_results:
        flagged = [row for row in anomaly_results if row.get("is_anomaly")]
        st.subheader(f"Flagged Real-Series Anomalies ({len(flagged)}/{len(anomaly_results)})")
        if flagged:
            st.dataframe(pd.DataFrame(flagged), use_container_width=True)

elif page == "🤖 LLM Assistant":
    st.title("LLM Analytics Assistant")
    st.caption("Grounded explanations from saved model outputs")

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
            "Ask a question:",
            value="Generate an executive summary based on all available model outputs.",
            height=90,
        )

        col1, col2 = st.columns([2, 3])
        with col1:
            provider = st.selectbox("Provider", ["mock", "anthropic"])
        with col2:
            model = st.text_input("Model", value="claude-3-5-haiku-latest")

        if st.button("Ask", type="primary"):
            with st.spinner("Querying..."):
                result = query_llm(
                    question=question,
                    context_str=context_str,
                    provider=provider,
                    model=model,
                    max_tokens=600,
                    temperature=0.2,
                )

            st.markdown(result["answer"])

            if result["success"]:
                st.caption(
                    f"Model: {result['model_used']} | "
                    f"Input tokens: {result['input_tokens']} | "
                    f"Output tokens: {result['output_tokens']}"
                )

        eval_filled = RESULTS_DIR / "llm_human_eval_filled.csv"
        if eval_filled.exists():
            st.divider()
            st.subheader("Human Evaluation Scores")

            eval_df = pd.read_csv(eval_filled)
            score_cols = ["accuracy_1_5", "usefulness_1_5", "groundedness_1_5", "clarity_1_5"]
            available = [c for c in score_cols if c in eval_df.columns]

            if available:
                numeric = eval_df[available].apply(pd.to_numeric, errors="coerce")
                metric_cols = st.columns(len(available))
                labels = ["Accuracy", "Usefulness", "Groundedness", "Clarity"]

                for col, label, score_col in zip(metric_cols, labels, available):
                    with col:
                        st.metric(label, f"{numeric[score_col].mean():.2f} / 5")