#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --job-name=hyb_bp_nsg_v10_gen
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-hyb-bp-nsg-gen-%j.out
set -euo pipefail
ROOT=/home/rsadve1/scratch/Neuro_Symbolic_GRAND
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
python -m hybrid_bp_nsg.cli --config configs/fir_hybrid_bp_nsg_full.yaml generate
bash scripts/status.sh outputs/hybrid_bp_nsg_v10_full
