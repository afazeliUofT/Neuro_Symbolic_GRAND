#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v13-smoke
#SBATCH --time=02:30:00
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /cvmfs/soft.computecanada.ca/config/profile/bash.sh /cvmfs/soft.computecanada.ca/config/profile/bash_modules.sh; do
    [ -r "$f" ] && source "$f"
  done
fi
module load cuda/12.6 2>/dev/null || true
if [ -d ".venv" ]; then source .venv/bin/activate; else echo "ERROR: .venv not found. Activate/create the FIR venv with TensorFlow/Sionna." >&2; exit 2; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TF_NUM_INTEROP_THREADS=4
python -m pip install -e . --no-deps --quiet
python scripts/check_tf_gpu.py --require-gpu
python scripts/check_config.py configs/fir_hybrid_bp_nsg_smoke.yaml
python scripts/check_channel_models.py --config configs/fir_hybrid_bp_nsg_smoke.yaml --profiles AWGN CDL_C
python -m hybrid_bp_nsg.cli pipeline --config configs/fir_hybrid_bp_nsg_smoke.yaml
