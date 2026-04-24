#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v12-gen
#SBATCH --time=24:00:00
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -d ".venv" ]; then source .venv/bin/activate; else echo "ERROR: .venv not found" >&2; exit 2; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
python -m pip install -e . --no-deps --quiet
python scripts/check_runtime_deps.py || true
python scripts/check_config.py configs/fir_hybrid_bp_nsg_full.yaml
python -m hybrid_bp_nsg.cli generate --config configs/fir_hybrid_bp_nsg_full.yaml
