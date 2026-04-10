# Causal Demand Forecasting & Decision Intelligence Platform

> A 7-module retail analytics platform that moves beyond standard demand forecasting into **causal decision intelligence** — built on the Corporación Favorita grocery dataset using University of Luxembourg HPC infrastructure.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-best_model-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io)
[![EconML](https://img.shields.io/badge/EconML-CausalForest-purple?style=flat-square)](https://econml.azurewebsites.net)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## What This Project Does

Most retail forecasting projects stop at predicting demand. This platform asks a harder question:

> **"What actually causes demand to change — and how much?"**

It combines a full forecasting stack with causal inference, price elasticity estimation, a computer-vision anomaly detector, and an LLM-powered analytics assistant into a single production-grade pipeline.

---

## Key Results

| Module | Result |
|--------|--------|
| **Forecasting** | LightGBM RMSE **17.39** vs baseline **24.42** — 29% improvement |
| **Probabilistic** | Coverage@90 = **0.8909** (target: 0.90) — well-calibrated intervals |
| **Causal Inference** | True promotion lift = **+3.15 units/day** (p=0.0004) — vs naive estimate of +13.67 |
| **Selection Bias** | Naive analysis **overestimates** promotion impact by **77%** |
| **Placebo Test** | **PASSED** — validates parallel trends assumption |
| **Elasticity** | Promotions increase demand by **+47–50%** on average (panel OLS, p≈0) |
| **Anomaly Detection** | ResNet-18 macro F1 = **0.9950** on synthetic holdout (4 classes) |
| **LLM Assistant** | 25 business questions answered and human-evaluated (Accuracy: 4.2/5) |

### System Progression

| Configuration | RMSE | Coverage@90 | Causal | Vision | LLM |
|---------------|------|-------------|--------|--------|-----|
| Baseline (Seasonal Naive) | 24.42 | — | No | No | No |
| + LightGBM Point | 17.39 | — | No | No | No |
| + Probabilistic Intervals | 19.50 | 0.8909 | No | No | No |
| + Causal Inference | 19.50 | 0.8909 | ATT=+3.15 | No | No |
| + Visual Anomaly Detector | 19.50 | 0.8909 | ATT=+3.15 | F1=0.995 | No |
| **Full System** | **19.50** | **0.8909** | **ATT=+3.15** | **F1=0.995** | **Yes** |

---

## Architecture

```
causal-demand-forecasting-platform/
│
├── Module 1  —  Data Pipeline          (src/data/, scripts/data_preparation.py)
├── Module 2  —  Demand Forecasting     (src/forecasting/, scripts/forecasting.py)
├── Module 3  —  Causal Inference       (src/causal/, scripts/causal_inference.py)
├── Module 4  —  Price Elasticity       (src/promotion_analysis/)
├── Module 5  —  Scenario Simulation    (src/simulation/, scripts/promotion_analysis.py)
├── Module 6  —  Visual Anomaly Det.    (src/anomaly_detection/, scripts/anomaly_detection.py)
└── Module 7  —  LLM Assistant          (src/llm/, scripts/llm_pipeline.py)

app/
└── streamlit_app.py    ← Interactive dashboard

slurm/                  ← HPC job scripts (University of Luxembourg)
configs/
└── base.yaml           ← Single config for the entire pipeline
```

---

## Technical Highlights

### Why Causal Inference, Not Just Forecasting

Standard forecasting tells you *what* demand will be. It cannot tell you *why*. A naive comparison of promoted vs non-promoted periods shows +13.67 units/day. But that estimate is contaminated by **selection bias** — retailers promote their best-selling products, so the promoted group already had higher baseline sales.

Difference-in-Differences controls for this by comparing the *change* in sales, not the level. The true causal estimate is **+3.15 units/day** — 77% lower than the naive estimate. This is the number a business should actually use when evaluating promotion ROI.

### Why Visual Anomaly Detection

Threshold-based anomaly detection (flag anything > 2σ) is brittle. It misses gradual structural breaks and generates false alarms on seasonal patterns. The ResNet-18 classifier learns the *visual shape* of anomalies from 8,000 synthetic chart images — exactly the spatial representation learning that Computer Vision research is built on.

This is also the only module in the project that directly applies a Computer Vision MSc specialisation to a domain problem. The Grad-CAM visualisations confirm the model focuses on the anomaly region of the chart, not background noise.

### Domain Shift Finding

On the synthetic holdout, the detector achieves macro F1 = 0.995. On real test-set series, 49/50 were flagged as anomalous. This is a documented **domain shift** problem: high-volume retail series contain inherent volatility (promotions, seasonality, stockouts) that visually resembles the spike/structural-break classes the model was trained to detect.

This is not a failure — it is an honest and important finding. The remedy is either:
1. Include real labelled examples in training
2. Calibrate the decision threshold to real-series volatility levels

---

## Dataset

**Corporación Favorita Grocery Sales Forecasting** (Kaggle)

- 125 million rows of daily sales data
- 54 stores, 4,000+ products, 5 years (2013–2017)
- Promotion flags, oil prices, holiday calendar, store and product metadata
- Source: [kaggle.com/competitions/favorita-grocery-sales-forecasting](https://kaggle.com/competitions/favorita-grocery-sales-forecasting)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/mnansari370/causal-demand-forecasting-platform
cd causal-demand-forecasting-platform

# 2. Create conda environment
conda create -n cdf_env python=3.11 -y
conda activate cdf_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Favorita data from Kaggle and place in:
#    data/raw/favorita/
```

**requirements.txt** (key packages):

```
lightgbm>=4.0
xgboost>=2.0
statsmodels>=0.14
linearmodels>=6.0
econml>=0.15
prophet>=1.1
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
streamlit>=1.32
anthropic>=0.25
PyYAML>=6.0
pyarrow>=14.0
tqdm>=4.65
Pillow>=10.0
```

---

## Run the Full Pipeline

All scripts read from `configs/base.yaml`. Run in order:

```bash
# Phase 1 — Data pipeline (45–90 min on HPC, ~5 min locally)
python scripts/create_dev_subset.py    # 2-pass store selection
python scripts/data_preparation.py    # clean, merge, split
python scripts/feature_engineering.py # lag + rolling features

# Phase 2 — Baseline
python scripts/baseline.py

# Phase 3 — Forecasting (60–120 min, trains LightGBM / XGBoost / Prophet / SARIMAX)
python scripts/forecasting.py

# Phase 4 — Causal inference (60–90 min, Causal Forest is slow)
python scripts/causal_inference.py

# Phase 5 — Promotion sensitivity and scenario simulation
python scripts/promotion_analysis.py

# Phase 6 — Generate synthetic dataset, then train anomaly detector (60–120 min)
python src/anomaly_detection/generate_synthetic_charts.py
python scripts/anomaly_detection.py

# Phase 7 — LLM pipeline (requires ANTHROPIC_API_KEY, ~$0.08 for 25 queries)
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/llm_pipeline.py

# Phase 8 — Streamlit dashboard
streamlit run app/streamlit_app.py
```

### HPC (SLURM)

Each stage has a corresponding SLURM job script in `slurm/`:

```bash
sbatch slurm/data_preparation.sh    # Phase 1
sbatch slurm/baseline.sh            # Phase 2
sbatch slurm/forecasting.sh         # Phase 3
sbatch slurm/causal_inference.sh    # Phase 4
sbatch slurm/promotion_analysis.sh  # Phase 5
sbatch slurm/anomaly_detection.sh   # Phase 6 (generates charts + trains model)
sbatch slurm/llm_pipeline.sh        # Phase 7
```

---

## Configuration

Everything is controlled from `configs/base.yaml`:

```yaml
splits:
  train_end:  "2016-12-31"   # strict temporal split — no data leakage
  val_start:  "2017-01-01"
  test_start: "2017-07-01"

cv_anomaly:
  n_images_per_class: 2000   # 4 classes × 2000 = 8000 images
  epochs: 10                 # 3 warmup + 7 fine-tune
  model_architecture: "resnet18"

llm:
  provider: "anthropic"      # set to "mock" for offline mode
  model_name: "claude-haiku-4-5"
```

---

## Module Details

### Module 1–2: Data Pipeline & Forecasting

- **Temporal split**: all splits are strictly chronological. No shuffling, no random splits.
- **Feature engineering**: lag features (t-1, t-7, t-14, t-28), rolling mean/std (7d, 28d), calendar features, promotion recency, smoothed target encoding.
- **Models**: LightGBM (best), XGBoost, LightGBM Quantile (Q0.05/0.5/0.95), Prophet, SARIMAX.
- **Evaluation**: RMSE, MAE, MAPE, Coverage@90, Interval Width.

### Module 3: Causal Inference

**Method**: Difference-in-Differences (DiD) with OLS regression.

```
unit_sales_mean ~ treated + post + treated:post
```

The coefficient on `treated:post` is the Average Treatment Effect on the Treated (ATT).

**Validation**: Placebo test applies the same DiD estimator to a pre-treatment window. A near-zero placebo estimate confirms the parallel trends assumption holds.

**Heterogeneous effects**: EconML `CausalForestDML` estimates individual treatment effects per item and store — answering "where should we run the next campaign?"

### Module 4–5: Elasticity & Simulation

- **Family-level OLS**: `log1p(demand) ~ onpromotion + controls` per product family
- **Panel OLS**: item fixed effects via `linearmodels.PanelOLS`
- **Scenario simulation**: combines LightGBM quantile predictions (baseline), panel coefficient (multiplicative effect), and DiD estimate (additive causal effect) — kept separate to avoid double-counting

### Module 6: Visual Anomaly Detector

**Architecture**: ResNet-18 pretrained on ImageNet, final layer replaced with a 4-class head (`Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→4)`).

**Training strategy**:
- Epochs 1–3: frozen backbone, only head trained (lr=1e-3)
- Epochs 4–10: full network fine-tuned (lr=1e-4)

**Synthetic dataset**: 8,000 images (2,000 per class) generated by programmatically injecting anomalies into smooth retail-like demand curves and rendering as 224×224 PNG charts.

**Explainability**: Grad-CAM implemented manually (no external library) using hooks on `model.layer4[-1]`.

### Module 7: LLM Analytics Assistant

**Design principle**: The LLM is the *presentation layer*, not the *reasoning layer*. All numbers come from saved model outputs loaded into a structured context block. The system prompt instructs Claude to ground every claim in the provided context and never fabricate metrics.

**Provider**: Anthropic Claude Haiku 4.5 (`claude-haiku-4-5`)

**Cost**: approximately $0.08 for the full 25-question evaluation run.

---

## Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

**Pages**:

| Page | Contents |
|------|----------|
| Overview | KPI summary, system architecture, key findings, results progression table |
| Forecasting | Model comparison table, RMSE bar chart, feature importance, calibration plot |
| Causal Inference | DiD results, selection bias decomposition, HTE store and item rankings |
| Promotion Analysis | Family sensitivity chart, panel results, scenario simulation |
| Anomaly Detector | Per-class F1, confusion matrix, training history, Grad-CAM gallery, flagged series |
| LLM Assistant | Chat interface, quick-prompt buttons, human evaluation scores |

**To run with SSH port forwarding from HPC**:

```bash
# On HPC
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true &

# On local machine
ssh -L 8501:localhost:8501 username@hpc.uni.lu
# Then open http://localhost:8501
```

---

## LLM Setup

```bash
# 1. Sign up at https://console.anthropic.com (free — $5 credit, no card needed)
# 2. Create an API key
# 3. Set the key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. The config already has provider: "anthropic"
# 5. Run the pipeline
python scripts/llm_pipeline.py
```

The pipeline generates `outputs/evaluation/llm_responses.json` (25 Q&A pairs) and `llm_human_eval.csv` (blank scoring template for manual evaluation).

---

## Project Structure

```
causal-demand-forecasting-platform/
│
├── app/
│   └── streamlit_app.py           ← Interactive Streamlit dashboard
│
├── configs/
│   └── base.yaml                  ← Single config for everything
│
├── data/
│   ├── raw/favorita/              ← Kaggle data (not tracked)
│   ├── interim/favorita/          ← Dev subset parquet
│   ├── processed/favorita/        ← train/val/test + features
│   └── synthetic/anomaly_charts/  ← 8,000 generated chart images
│
├── outputs/
│   ├── evaluation/                ← All JSON/CSV results
│   ├── figures/                   ← All PNG figures
│   └── models/                   ← Trained model files
│
├── scripts/                       ← One script per pipeline stage
│   ├── create_dev_subset.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── baseline.py
│   ├── forecasting.py
│   ├── causal_inference.py
│   ├── promotion_analysis.py
│   ├── anomaly_detection.py
│   └── llm_pipeline.py
│
├── slurm/                         ← HPC SLURM job scripts
│
└── src/                           ← Importable source modules
    ├── anomaly_detection/         ← ResNet-18 train/eval/inference + Grad-CAM
    ├── causal/                    ← DiD, placebo, Causal Forest
    ├── data/                      ← Loading, cleaning, preprocessing
    ├── evaluation/                ← Metrics and evaluation utilities
    ├── features/                  ← Feature engineering
    ├── forecasting/               ← LightGBM, XGBoost, Prophet, SARIMAX
    ├── llm/                       ← Context builder + Anthropic assistant
    ├── promotion_analysis/        ← Family OLS, panel OLS, revenue proxy
    ├── simulation/                ← Scenario simulation engine
    └── utils/                     ← Logger
```

---

## Environment

Built and tested on:
- **HPC**: University of Luxembourg Aion cluster (SLURM, batch partition)
- **Python**: 3.11, conda environment
- **GPU**: NVIDIA (batch partition, used for ResNet-18 fine-tuning)
- **Dataset**: Corporación Favorita (4.7 GB raw train.csv, 125M rows)

---

## What Makes This Different From a Standard Forecasting Project

1. **Causal layer with validation** — DiD + placebo test + Causal Forest. Almost no student project includes the placebo test. It demonstrates methodological maturity.

2. **Quantified selection bias** — showing *exactly* how much naive business reporting overestimates promotion effectiveness (77% in this case) is a concrete finding with immediate business value.

3. **CV specialisation applied to a domain problem** — using ResNet-18 for time-series chart classification is a direct application of computer vision to a non-vision domain. The Grad-CAM explainability layer makes it interpretable.

4. **End-to-end pipeline on real HPC infrastructure** — not a notebook. A production-style pipeline with configuration management, logging, SLURM scripts, and reproducible splits.

5. **LLM as a presentation layer, not a reasoning layer** — the design principle ensures all numbers come from real model outputs. The LLM translates, not fabricates.

---

## Citation / Acknowledgements

Dataset: Corporación Favorita Grocery Sales Forecasting, Kaggle (2017).

Libraries used: LightGBM, XGBoost, Prophet, statsmodels, linearmodels, EconML, PyTorch, torchvision, Streamlit, Anthropic Python SDK, pandas, numpy, matplotlib, scikit-learn.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of MSc Computer Science (AI/CV specialisation), University of Luxembourg.*
