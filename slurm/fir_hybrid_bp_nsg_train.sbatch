#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --job-name=hyb_bp_nsg_v10_train
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-hyb-bp-nsg-train-%j.out
set -euo pipefail
ROOT=/home/rsadve1/scratch/Neuro_Symbolic_GRAND
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
python -m hybrid_bp_nsg.cli --config configs/fir_hybrid_bp_nsg_full.yaml train
bash scripts/status.sh outputs/hybrid_bp_nsg_v10_full
