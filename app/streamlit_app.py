"""
Streamlit demo interface for the demand forecasting platform.

Run with:
    streamlit run app/streamlit_app.py

The app reads from precomputed outputs in outputs/evaluation/ and
outputs/figures/. All heavy computation happens in the scripts, not
inside the app itself. This keeps the interface fast and reproducible.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Causal Demand Forecasting Platform",
    page_icon="📊",
    layout="wide",
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        section[data-testid="stSidebar"] {
            min-width: 260px;
            max-width: 260px;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }

        .soft-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .section-note {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 0.4rem;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .chat-tip {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 14px;
        }

        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 10px 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data loaders ----------
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


@st.cache_data
def get_image_size(path: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


# ---------- Helpers ----------
def fmt_number(value, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return "N/A"
        if digits == 0:
            return f"{int(round(val))}{suffix}"
        return f"{val:.{digits}f}{suffix}"
    except Exception:
        return str(value)


def safe_metric(label: str, value, digits: int = 4, suffix: str = "") -> None:
    st.metric(label, fmt_number(value, digits=digits, suffix=suffix))


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <div style="font-weight:700; margin-bottom:6px;">{title}</div>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_header() -> None:
    st.title("Causal Demand Forecasting & Decision Intelligence Platform")
    st.markdown(
        """
        <div class="app-subtitle">
            Forecasting, causal inference, promotion sensitivity, anomaly detection,
            and an LLM-based analytics assistant over precomputed retail analytics outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_figure(
    path: Path,
    caption: str = "",
    max_width: int = 850,
    center: bool = True,
) -> None:
    if not path.exists():
        st.info(f"Figure not found: {path.name}. Run the corresponding pipeline first.")
        return

    size = get_image_size(str(path))
    width = max_width
    if size is not None:
        width = min(max_width, size[0])

    if center:
        left, mid, right = st.columns([1, 8, 1])
        with mid:
            st.image(str(path), caption=caption, width=width)
    else:
        st.image(str(path), caption=caption, width=width)


def show_small_figure(path: Path, caption: str = "") -> None:
    show_figure(path, caption=caption, max_width=520, center=False)


def sidebar_status() -> None:
    forecasting = load_json(RESULTS_DIR / "forecasting_results.json")
    did = load_json(RESULTS_DIR / "causal_did_result.json")
    panel = load_json(RESULTS_DIR / "panel_promotion_sensitivity.json")
    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")
    llm_responses = load_json(RESULTS_DIR / "llm_responses.json")

    st.sidebar.markdown("### Module Status")
    st.sidebar.success("Forecasting ready" if forecasting else "Forecasting missing")
    st.sidebar.success("Causal ready" if did else "Causal missing")
    st.sidebar.success("Promotion ready" if panel else "Promotion missing")
    st.sidebar.success("Anomaly ready" if cv_eval else "Anomaly missing")
    st.sidebar.success("LLM results ready" if llm_responses else "LLM results missing")
    st.sidebar.caption("The assistant answers from saved outputs, not raw data.")


def build_kpi_summary():
    forecasting = load_json(RESULTS_DIR / "forecasting_results.json")
    did = load_json(RESULTS_DIR / "causal_did_result.json")
    panel = load_json(RESULTS_DIR / "panel_promotion_sensitivity.json")
    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")

    naive_rmse = None
    lgbm_rmse = None
    cov90 = None

    if forecasting:
        naive = next((r for r in forecasting if "Naive" in str(r.get("model", ""))), {})
        lgbm = next((r for r in forecasting if r.get("model") == "LightGBM Point"), {})
        quant = next((r for r in forecasting if "Quantile" in str(r.get("model", ""))), {})
        naive_rmse = naive.get("rmse")
        lgbm_rmse = lgbm.get("rmse")
        cov90 = quant.get("coverage_90")

    return {
        "naive_rmse": naive_rmse,
        "lgbm_rmse": lgbm_rmse,
        "cov90": cov90,
        "did_estimate": did.get("estimate") if did else None,
        "did_p": did.get("p_value") if did else None,
        "panel_uplift": panel.get("pct_demand_change") if panel else None,
        "macro_f1": cv_eval.get("macro_f1") if cv_eval else None,
    }


# ---------- Pages ----------
def render_overview() -> None:
    app_header()
    summary = build_kpi_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        safe_metric("Baseline RMSE", summary["naive_rmse"])
    with c2:
        safe_metric("LightGBM RMSE", summary["lgbm_rmse"])
    with c3:
        safe_metric("Coverage@90", summary["cov90"])
    with c4:
        safe_metric("DiD Promotion Lift", summary["did_estimate"], suffix=" u/day")

    if summary["naive_rmse"] is not None and summary["lgbm_rmse"] is not None:
        try:
            improvement = (1 - float(summary["lgbm_rmse"]) / float(summary["naive_rmse"])) * 100
            st.success(
                f"LightGBM reduced RMSE by **{improvement:.1f}%** compared to the seasonal naive baseline."
            )
        except Exception:
            pass

    left, right = st.columns([1.15, 1])

    with left:
        section_header("What this platform does")
        info_card(
            "Forecasting",
            "Predicts future demand and quantifies uncertainty with point and quantile forecasts.",
        )
        info_card(
            "Causal Inference",
            "Estimates the true effect of promotions instead of relying on biased before-after comparisons.",
        )
        info_card(
            "Promotion Sensitivity",
            "Shows which items and families respond most strongly to promotions.",
        )
        info_card(
            "Anomaly Detection",
            "Flags unusual demand patterns using a visual classifier over chart images.",
        )
        info_card(
            "LLM Assistant",
            "Turns saved analytical outputs into natural-language answers for business users.",
        )

    with right:
        section_header("Headline results")
        info_card(
            "Forecasting quality",
            f"Best point forecast RMSE: {fmt_number(summary['lgbm_rmse'])}. Coverage@90: {fmt_number(summary['cov90'])}.",
        )
        info_card(
            "Causal finding",
            f"Promotions increased demand by about {fmt_number(summary['did_estimate'])} units/day "
            f"with p-value {fmt_number(summary['did_p'])}.",
        )
        info_card(
            "Promotion uplift",
            f"Panel regression suggests an average demand uplift of {fmt_number(summary['panel_uplift'], digits=1, suffix='%')}.",
        )
        info_card(
            "Anomaly detector",
            f"Synthetic test macro F1: {fmt_number(summary['macro_f1'])}.",
        )

    st.divider()

    table_path = RESULTS_DIR / "main_results_table.txt"
    if table_path.exists():
        section_header("Main results table")
        st.code(table_path.read_text(encoding="utf-8"), language=None)
    else:
        st.info("Run scripts/llm_pipeline.py to generate the main results table.")


def render_forecasting() -> None:
    st.title("Forecasting")

    forecasting = load_json(RESULTS_DIR / "forecasting_results.json")
    if forecasting:
        df = pd.DataFrame(forecasting)
        cols = [
            c
            for c in [
                "model",
                "rmse",
                "mae",
                "mape",
                "coverage_90",
                "interval_width",
                "n_samples",
            ]
            if c in df.columns
        ]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

        lgbm = next((r for r in forecasting if r.get("model") == "LightGBM Point"), {})
        quant = next((r for r in forecasting if "Quantile" in str(r.get("model", ""))), {})
        xgb = next((r for r in forecasting if "XGBoost" in str(r.get("model", ""))), {})

        c1, c2, c3 = st.columns(3)
        with c1:
            safe_metric("LightGBM RMSE", lgbm.get("rmse"))
        with c2:
            safe_metric("Quantile Coverage@90", quant.get("coverage_90"))
        with c3:
            safe_metric("XGBoost RMSE", xgb.get("rmse"))
    else:
        st.info("No forecasting results found. Run scripts/forecasting.py.")
        return

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        show_small_figure(FIGURES_DIR / "lgbm_feature_importance.png", "LightGBM Feature Importance")
    with col2:
        show_small_figure(FIGURES_DIR / "lgbm_quantile_calibration.png", "Quantile Calibration")

    show_figure(
        FIGURES_DIR / "sample_quantile_forecast.png",
        "Sample forecast with 90% prediction interval",
        max_width=920,
    )


def render_causal() -> None:
    st.title("Causal Inference")

    did = load_json(RESULTS_DIR / "causal_did_result.json")
    placebo = load_json(RESULTS_DIR / "causal_placebo_result.json")
    naive = load_json(RESULTS_DIR / "causal_naive_comparison.json")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        safe_metric("Naive estimate", naive.get("naive_estimate") if naive else None, suffix=" u/day")
    with c2:
        safe_metric("DiD ATT", did.get("estimate") if did else None, suffix=" u/day")
    with c3:
        safe_metric("p-value", did.get("p_value") if did else None)
    with c4:
        if naive and did and naive.get("naive_estimate"):
            try:
                bias = abs((naive["naive_estimate"] - did["estimate"]) / naive["naive_estimate"] * 100)
                safe_metric("Selection bias", bias, digits=1, suffix="%")
            except Exception:
                safe_metric("Selection bias", None)
        else:
            safe_metric("Selection bias", None)

    if placebo:
        verdict = placebo.get("verdict", "N/A")
        if placebo.get("passed") is True:
            st.success(f"Placebo test: {verdict}")
        elif placebo.get("passed") is False:
            st.error(f"Placebo test: {verdict}")
        else:
            st.info(f"Placebo test: {verdict}")

    show_figure(FIGURES_DIR / "causal_did_summary.png", "Naive vs causal vs placebo estimate", max_width=860)

    col1, col2 = st.columns(2)
    with col1:
        show_small_figure(FIGURES_DIR / "causal_store_hte.png", "Store-level heterogeneous treatment effects")
    with col2:
        show_small_figure(FIGURES_DIR / "causal_item_hte.png", "Item-level heterogeneous treatment effects")

    item_hte = load_csv(RESULTS_DIR / "causal_item_hte.csv")
    if item_hte is not None:
        section_header("Top item-level heterogeneous effects")
        st.dataframe(item_hte.head(20), use_container_width=True, hide_index=True)


def render_promotion() -> None:
    st.title("Promotion Sensitivity")

    panel = load_json(RESULTS_DIR / "panel_promotion_sensitivity.json")
    if panel:
        c1, c2, c3 = st.columns(3)
        with c1:
            safe_metric("Promotion coefficient", panel.get("promotion_coef"))
        with c2:
            safe_metric("Demand uplift", panel.get("pct_demand_change"), digits=1, suffix="%")
        with c3:
            safe_metric("Items analysed", panel.get("n_items"), digits=0)

    show_figure(
        FIGURES_DIR / "promotion_sensitivity_by_family.png",
        "Promotion sensitivity by product family",
        max_width=860,
    )

    scenario_df = load_csv(RESULTS_DIR / "scenario_grid.csv")
    if scenario_df is not None:
        section_header("Scenario grid")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        show_small_figure(FIGURES_DIR / "simulation_panel.png", "Panel-based promotion scenario")
    with col2:
        show_small_figure(FIGURES_DIR / "simulation_did.png", "DiD-based promotion scenario")


def render_anomaly() -> None:
    st.title("Anomaly Detector")

    cv_eval = load_json(RESULTS_DIR / "cv_evaluation_results.json")
    if cv_eval:
        c1, c2, c3 = st.columns(3)
        with c1:
            safe_metric("Accuracy", cv_eval.get("accuracy"))
        with c2:
            safe_metric("Macro F1", cv_eval.get("macro_f1"))
        with c3:
            safe_metric("Test samples", cv_eval.get("n_test_samples"), digits=0)

        per_class = cv_eval.get("per_class_metrics") or {}
        if per_class:
            section_header("Per-class metrics")
            st.dataframe(
                pd.DataFrame([{"class": cls, **metrics} for cls, metrics in per_class.items()]),
                use_container_width=True,
                hide_index=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        show_small_figure(FIGURES_DIR / "cv_confusion_matrix.png", "Confusion matrix")
    with col2:
        show_small_figure(FIGURES_DIR / "cv_per_class_metrics.png", "Per-class metrics")

    section_header("Grad-CAM examples")
    gradcam_files = sorted(FIGURES_DIR.glob("gradcam_*.png"))
    if gradcam_files:
        cols = st.columns(3)
        for i, img in enumerate(gradcam_files[:9]):
            with cols[i % 3]:
                show_figure(
                    img,
                    img.stem.replace("_", " ").title(),
                    max_width=320,
                    center=False,
                )
    else:
        st.info("No Grad-CAM images found.")

    anomaly_results = load_json(RESULTS_DIR / "anomaly_detection_results.json")
    if anomaly_results:
        flagged = [row for row in anomaly_results if row.get("is_anomaly")]
        section_header(f"Flagged real-series anomalies ({len(flagged)}/{len(anomaly_results)})")
        if flagged:
            st.dataframe(pd.DataFrame(flagged), use_container_width=True, hide_index=True)


def render_llm() -> None:
    st.title("LLM Analytics Assistant")
    st.caption("Ask natural-language questions grounded in saved model outputs.")

    try:
        from src.llm.assistant import query_llm
        from src.llm.context_builder import build_context, format_context_for_prompt

        context = build_context(RESULTS_DIR)
        context_str = format_context_for_prompt(context)
        llm_ready = True
    except Exception as exc:
        st.error(f"Failed to load LLM modules: {exc}")
        llm_ready = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    sample_prompts = [
        "Did promotions really work?",
        "What are the top 3 business insights from this analysis?",
        "Which forecasting model performed best and why?",
        "Why is the naive promotion estimate misleading?",
        "Why were 49 out of 50 real series flagged as anomalies?",
        "What should a retail manager do next quarter based on these results?",
    ]

    st.markdown(
        """
        <div class="chat-tip">
            Ask questions about forecasting, causal inference, promotion sensitivity,
            anomaly detection, or overall business insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_cols = st.columns(len(sample_prompts))
    for i, prompt in enumerate(sample_prompts):
        with prompt_cols[i]:
            if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                st.session_state.pending_prompt = prompt

    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        pending_prompt = st.session_state.pending_prompt
    else:
        pending_prompt = None

    if llm_ready:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("meta"):
                    st.caption(message["meta"])

        user_question = st.chat_input(
            "Ask something about your project results..."
        )

        if pending_prompt and not user_question:
            user_question = pending_prompt
            st.session_state.pending_prompt = None

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query_llm(
                        question=user_question,
                        context_str=context_str,
                        provider=DEFAULT_PROVIDER,
                        model=DEFAULT_MODEL,
                        max_tokens=600,
                        temperature=0.2,
                    )

                st.markdown(result["answer"])

                meta = ""
                if result.get("success"):
                    meta = (
                        f"Model: {result.get('model_used', DEFAULT_MODEL)} | "
                        f"Input tokens: {result.get('input_tokens', 0)} | "
                        f"Output tokens: {result.get('output_tokens', 0)}"
                    )
                    st.caption(meta)
                else:
                    meta = "The assistant could not generate a successful response."
                    st.caption(meta)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": meta,
                }
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            if st.button("Show example questions", use_container_width=True):
                st.info(
                    "Examples: Did promotions really work? | Which model performed best? | "
                    "Why is the naive estimate misleading? | What are the top 3 business insights?"
                )

        eval_filled = RESULTS_DIR / "llm_human_eval_filled.csv"
        if eval_filled.exists():
            st.divider()
            section_header("Human evaluation scores")

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


# ---------- App shell ----------
sidebar_status()

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
    render_overview()
elif page == "📈 Forecasting":
    render_forecasting()
elif page == "🔬 Causal Inference":
    render_causal()
elif page == "💰 Promotion Sensitivity":
    render_promotion()
elif page == "👁️ Anomaly Detector":
    render_anomaly()
elif page == "🤖 LLM Assistant":
    render_llm()