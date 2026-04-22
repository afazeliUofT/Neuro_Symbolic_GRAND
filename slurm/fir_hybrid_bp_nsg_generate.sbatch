#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v112-gen
#SBATCH --time=24:00:00
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -d ".venv" ]; then source .venv/bin/activate; else echo "ERROR: .venv not found" >&2; exit 2; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -e . --no-deps --quiet
python scripts/check_runtime_deps.py || true
python -m hybrid_bp_nsg.cli generate --config configs/fir_hybrid_bp_nsg_full.yaml
