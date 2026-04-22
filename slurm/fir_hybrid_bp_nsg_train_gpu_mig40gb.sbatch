#!/usr/bin/env bash
#SBATCH --job-name=hyb-grand-v112-train-mig40
#SBATCH --time=24:00:00
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if ! type module >/dev/null 2>&1; then
  for f in \
    /etc/profile.d/modules.sh \
    /etc/profile.d/lmod.sh \
    /cvmfs/soft.computecanada.ca/config/profile/bash.sh \
    /cvmfs/soft.computecanada.ca/config/profile/bash_modules.sh
  do
    [ -r "$f" ] && source "$f"
  done
fi

module load cuda/12.6 2>/dev/null || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found. Activate/create the same FIR venv that contains TensorFlow 2.19.1 and Sionna 1.2.2." >&2
  exit 2
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="${XLA_FLAGS:-}"

python -m pip install -e . --no-deps --quiet
python scripts/check_tf_gpu.py --require-gpu

python -m hybrid_bp_nsg.cli train --config configs/fir_hybrid_bp_nsg_full.yaml
