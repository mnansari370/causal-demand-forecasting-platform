#!/bin/bash
#SBATCH --job-name=cdf_week4
#SBATCH --output=logs/slurm_week4_%j.log
#SBATCH --error=logs/slurm_week4_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job start: $(date)"
echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Week 4: Promotion sensitivity + scenario simulation ---"
python scripts/run_elasticity_simulation.py

echo "Job end: $(date)"
