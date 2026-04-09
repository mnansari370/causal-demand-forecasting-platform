#!/bin/bash
#SBATCH --job-name=cdf_baseline
#SBATCH --output=logs/slurm_baseline_%j.log
#SBATCH --error=logs/slurm_baseline_%j.err
#SBATCH --time=01:00:00
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

python scripts/baseline.py

echo "Job end: $(date)"