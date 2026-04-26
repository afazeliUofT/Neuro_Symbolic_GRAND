#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v13-selftest
#SBATCH --time=00:30:00
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -d ".venv" ]; then source .venv/bin/activate; else echo "ERROR: .venv not found" >&2; exit 2; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -e . --no-deps --quiet
python scripts/check_runtime_deps.py || true
python scripts/check_config.py configs/fir_hybrid_bp_nsg_selftest.yaml
python scripts/check_channel_models.py --config configs/fir_hybrid_bp_nsg_selftest.yaml --profiles AWGN
python -m hybrid_bp_nsg.cli selftest --config configs/fir_hybrid_bp_nsg_selftest.yaml
