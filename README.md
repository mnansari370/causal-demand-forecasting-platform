# Causal Demand Forecasting & Decision Intelligence Platform

## Overview
A retail analytics platform that combines demand forecasting, causal inference, promotion sensitivity analysis, anomaly detection, and an LLM-based analytics assistant on the Favorita dataset.

## Key Features
- Point and probabilistic demand forecasting
- Causal promotion impact estimation using Difference-in-Differences
- Promotion sensitivity analysis at item/family level
- Visual anomaly detection on time-series charts
- LLM assistant for natural-language business insights

## Tech Stack
- Python
- Pandas, NumPy
- LightGBM, XGBoost
- Statsmodels, EconML, linearmodels
- PyTorch, torchvision
- Streamlit
- Anthropic Claude API
- SLURM / HPC

## Project Architecture
Describe the full pipeline:
1. Data preparation
2. Feature engineering
3. Forecasting
4. Causal inference
5. Promotion sensitivity
6. Anomaly detection
7. LLM assistant
8. Streamlit demo

## Dataset
Corporación Favorita Grocery Sales Forecasting dataset.

## Results
- Baseline RMSE: 24.4195
- LightGBM RMSE: 17.3933
- Coverage@90: 0.8909
- DiD promotion lift: +3.149 units/day (p=0.0004)
- Panel promotion uplift: +50.0%
- Synthetic anomaly detector macro F1: 0.995
- Real-series anomaly detection flagged 49/50 high-volume series, indicating domain shift

## Example Questions the LLM Assistant Can Answer
- Did promotions really work?
- Which forecasting model performed best and why?
- Why is the naive promotion estimate misleading?
- What are the top 3 business insights from this analysis?

## Streamlit Demo
Explain how to run:
```bash
streamlit run app/streamlit_app.py