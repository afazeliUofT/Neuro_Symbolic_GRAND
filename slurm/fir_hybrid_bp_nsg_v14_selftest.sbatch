#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v14-selftest
#SBATCH --time=0:30:00
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
python scripts/check_package_version.py
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="-1"
export NVIDIA_VISIBLE_DEVICES="none"
python scripts/check_config.py configs/fir_hybrid_bp_nsg_selftest_v14.yaml
python -m hybrid_bp_nsg.cli selftest --config configs/fir_hybrid_bp_nsg_selftest_v14.yaml
