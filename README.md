# Causal Demand Forecasting & Decision Intelligence Platform

This project builds an end-to-end retail analytics system covering:

- demand forecasting
- causal inference for promotions
- scenario simulation and decision intelligence
- visual anomaly detection on forecast charts
- LLM-based analytics explanations
- Streamlit dashboard delivery

## Dataset

Primary dataset: Corporación Favorita Grocery Sales Forecasting (Kaggle)

## Project Modules

1. Data ingestion and feature engineering
2. Forecasting models
3. Causal inference
4. Elasticity and scenario simulation
5. CV anomaly detection
6. LLM analytics assistant
7. Dashboard and deployment interface

## Repository Structure

- `configs/` configuration files
- `data/` raw, interim, processed, synthetic data
- `src/` source code by module
- `scripts/` executable scripts
- `slurm/` HPC job scripts
- `notebooks/` exploratory work
- `app/` Streamlit app
- `outputs/` generated artifacts
- `logs/` run logs

## Environment

This project uses a Conda environment defined in `environment.yml`.

## Current Status

Foundation setup complete:
- environment
- GitHub
- dataset download
- data loading and preprocessing scaffold
- logging and reproducibility base