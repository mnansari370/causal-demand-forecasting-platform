#!/bin/bash
#SBATCH --job-name=cdf_causal
#SBATCH --output=logs/slurm_causal_%j.log
#SBATCH --error=logs/slurm_causal_%j.err
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

python scripts/causal_inference.py

echo "Job end: $(date)"