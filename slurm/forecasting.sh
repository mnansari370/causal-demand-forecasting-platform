#!/bin/bash
#SBATCH --job-name=cdf_forecasting
#SBATCH --output=logs/slurm_forecasting_%j.log
#SBATCH --error=logs/slurm_forecasting_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

set -e

echo "Job start: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

python scripts/forecasting.py

echo "Job end: $(date)"