#!/bin/bash
#SBATCH --job-name=fir-tf-gpu-probe
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/rsadve1/scratch/Neuro_Symbolic_GRAND/fir_tf_gpu_probe_20260421_195825/gpu_probe_%j.out

set +e
cd "/home/rsadve1/scratch/Neuro_Symbolic_GRAND"

echo "================================================================================"
echo "[BEGIN gpu_job_header]"
echo "================================================================================"
date
hostname
pwd
echo "USER=$USER"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_GPUS=${SLURM_GPUS:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "================================================================================"
echo "[END gpu_job_header]"
echo "================================================================================"

if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /cvmfs/soft.computecanada.ca/config/profile/bash.sh /cvmfs/soft.computecanada.ca/config/profile/bash_modules.sh; do
    [ -r "$f" ] && source "$f"
  done
fi

echo
echo "================================================================================"
echo "[BEGIN module_list_gpu_job]"
echo "================================================================================"
module list 2>&1 || true
echo "================================================================================"
echo "[END module_list_gpu_job]"
echo "================================================================================"

echo
echo "================================================================================"
echo "[BEGIN nvidia_smi_gpu_job]"
echo "================================================================================"
nvidia-smi 2>&1 || true
echo "================================================================================"
echo "[END nvidia_smi_gpu_job]"
echo "================================================================================"

source .venv/bin/activate 2>/dev/null || true

echo
echo "================================================================================"
echo "[BEGIN python_runtime_probe_gpu_job]"
echo "================================================================================"
which python
python -V
python "/home/rsadve1/scratch/Neuro_Symbolic_GRAND/fir_tf_gpu_probe_20260421_195825/py_runtime_probe.py"
echo "================================================================================"
echo "[END python_runtime_probe_gpu_job]"
echo "================================================================================"
