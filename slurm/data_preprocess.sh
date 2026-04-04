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

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

python scripts/build_dev_subset.py
python scripts/run_data_check.py
python scripts/build_forecasting_features.py

echo "Job end: $(date)"