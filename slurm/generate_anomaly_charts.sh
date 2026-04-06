#!/bin/bash
#SBATCH --job-name=cdf_anomaly_charts
#SBATCH --output=logs/slurm_anomaly_charts_%j.log
#SBATCH --error=logs/slurm_anomaly_charts_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job start: $(date)"
echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

python src/cv/generate_anomaly_charts.py

echo "Job end: $(date)"