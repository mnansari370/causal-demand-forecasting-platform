#!/bin/bash
#SBATCH --job-name=cdf_cv_week5
#SBATCH --output=logs/slurm_cv_%j.log
#SBATCH --error=logs/slurm_cv_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job start: $(date)"
echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Torch / CUDA check ---"
python - << 'EOF'
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
EOF

echo "--- Step 1: Generate synthetic anomaly charts ---"
python src/cv/generate_anomaly_charts.py

echo "--- Step 2: Train + evaluate + Grad-CAM + inference ---"
python scripts/run_cv_pipeline.py

echo "Job end: $(date)"