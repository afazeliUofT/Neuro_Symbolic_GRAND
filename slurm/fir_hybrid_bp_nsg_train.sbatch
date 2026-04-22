#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v11
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -d ".venv" ]; then source .venv/bin/activate; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -e . --no-deps --quiet
python -m hybrid_bp_nsg.cli train --config configs/fir_hybrid_bp_nsg_full.yaml
