#!/bin/bash
#SBATCH --job-name=cdf_data_prep
#SBATCH --output=logs/slurm_data_prep_%j.log
#SBATCH --error=logs/slurm_data_prep_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job start: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Step 1: Build development subset ---"
python scripts/build_dev_subset.py

echo "--- Step 2: Run data check and preprocessing ---"
python scripts/run_data_check.py

echo "--- Step 3: Build forecasting features ---"
python scripts/build_forecasting_features.py

echo "--- Step 4: Run baseline ---"
python scripts/run_baseline.py

echo "Job end: $(date)"