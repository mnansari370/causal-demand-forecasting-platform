#!/bin/bash
#SBATCH --job-name=cdf_anomaly
#SBATCH --output=logs/slurm_anomaly_%j.log
#SBATCH --error=logs/slurm_anomaly_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -e

echo "Job start: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Step 1: Generate synthetic charts ---"
python -m src.anomaly_detection.generate_synthetic_charts

echo "--- Step 2: Train, evaluate, and run inference ---"
python scripts/anomaly_detection.py

echo "Job end: $(date)"