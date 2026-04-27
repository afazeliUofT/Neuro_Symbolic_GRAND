#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v14-eval
#SBATCH --time=20:00:00
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
python scripts/check_package_version.py
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTEROP_THREADS=8
python scripts/check_tf_gpu.py --require-gpu
python scripts/check_config.py configs/fir_hybrid_bp_nsg_full_v14.yaml
python scripts/check_channel_models.py --config configs/fir_hybrid_bp_nsg_full_v14.yaml --profiles AWGN CDL_C
python -m hybrid_bp_nsg.cli evaluate --config configs/fir_hybrid_bp_nsg_full_v14.yaml
python -m hybrid_bp_nsg.cli report --config configs/fir_hybrid_bp_nsg_full_v14.yaml
