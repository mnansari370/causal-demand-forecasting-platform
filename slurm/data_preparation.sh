#!/bin/bash
#SBATCH --job-name=cdf_data_prep
#SBATCH --output=logs/slurm_data_prep_%j.log
#SBATCH --error=logs/slurm_data_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

set -e

echo "Job start: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Step 1: Create development subset ---"
python scripts/create_dev_subset.py

echo "--- Step 2: Prepare data ---"
python scripts/data_preparation.py

echo "--- Step 3: Build forecasting features ---"
python scripts/feature_engineering.py

echo "Job end: $(date)"