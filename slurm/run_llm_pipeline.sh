#!/bin/bash
#SBATCH --job-name=cdf_llm
#SBATCH --output=logs/slurm_llm_%j.log
#SBATCH --error=logs/slurm_llm_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch

set -e

echo "Job start: $(date)"
echo "Node:   $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate cdf_env
cd ~/causal-demand-forecasting-platform

echo "--- Environment check ---"
python - << 'EOF'
import os
print("ANTHROPIC_API_KEY set:", bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()))
EOF

echo "--- Week 6: LLM analytics pipeline ---"
python scripts/run_llm_pipeline.py

echo "Job end: $(date)"