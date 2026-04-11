#!/bin/bash
#SBATCH --job-name=nsgrand_v2_report
#SBATCH --account=def-rsadve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python -m neuro_symbolic_grand.cli --config configs/fir_v2_full.yaml report
