#!/bin/bash
#SBATCH --job-name=nsgrand_full
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

python -m neuro_symbolic_grand.cli --config configs/fir_full.yaml pipeline
