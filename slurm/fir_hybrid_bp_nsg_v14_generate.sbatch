#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v14-gen
#SBATCH --time=12:00:00
#SBATCH --partition=cpubase_bycore_b3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
python scripts/check_runtime_deps.py || true
python scripts/check_config.py configs/fir_hybrid_bp_nsg_full_v14.yaml
python scripts/check_channel_models.py --config configs/fir_hybrid_bp_nsg_full_v14.yaml --profiles AWGN CDL_C
python -m hybrid_bp_nsg.cli generate --config configs/fir_hybrid_bp_nsg_full_v14.yaml
python scripts/probe_v14_dataset.py --output outputs/hybrid_bp_nsg_v14_pusch_cdl_full
