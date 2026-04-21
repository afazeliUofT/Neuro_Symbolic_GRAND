#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --job-name=hyb_bp_nsg_v10_selftest
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hyb-bp-nsg-selftest-%j.out

set -euo pipefail
ROOT=/home/rsadve1/scratch/Neuro_Symbolic_GRAND
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
python -m hybrid_bp_nsg.cli --config configs/fir_hybrid_bp_nsg_selftest.yaml pipeline
"$ROOT/scripts/verify_run_complete.sh" "$ROOT/outputs/hybrid_bp_nsg_v10_selftest"
