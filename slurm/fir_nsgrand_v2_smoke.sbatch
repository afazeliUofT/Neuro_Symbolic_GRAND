#!/bin/bash
#SBATCH --job-name=nsgrand_v2_smoke
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

python -m neuro_symbolic_grand.cli --config configs/fir_v2_smoke.yaml pipeline
