#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --job-name=hyb_bp_nsg_v10_eval
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-hyb-bp-nsg-eval-%j.out
set -euo pipefail
ROOT=/home/rsadve1/scratch/Neuro_Symbolic_GRAND
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
python -m hybrid_bp_nsg.cli --config configs/fir_hybrid_bp_nsg_full.yaml evaluate
python -m hybrid_bp_nsg.cli --config configs/fir_hybrid_bp_nsg_full.yaml report
bash scripts/verify_run_complete.sh outputs/hybrid_bp_nsg_v10_full
