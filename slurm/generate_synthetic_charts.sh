#!/bin/bash
#SBATCH --job-name=cdf_synthetic_charts
#SBATCH --output=logs/slurm_synthetic_charts_%j.log
#SBATCH --error=logs/slurm_synthetic_charts_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -e

echo "Job start: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

# Generate the 8000 synthetic chart images used to train the anomaly detector
python src/anomaly_detection/generate_synthetic_charts.py

echo "Job end: $(date)"