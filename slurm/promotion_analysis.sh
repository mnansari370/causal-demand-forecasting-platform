#!/bin/bash
#SBATCH --job-name=cdf_promotion
#SBATCH --output=logs/slurm_promotion_%j.log
#SBATCH --error=logs/slurm_promotion_%j.err
#SBATCH --time=02:00:00
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

python scripts/promotion_analysis.py

echo "Job end: $(date)"