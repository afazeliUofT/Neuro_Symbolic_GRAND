# FIR command flow — v14.1

After unzipping this package directly inside `/home/rsadve1/scratch/Neuro_Symbolic_GRAND`:

```bash
cd ~/scratch/Neuro_Symbolic_GRAND
bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
python -m pip install -e . --no-deps
```

Recommended run order:

```bash
sbatch slurm/fir_hybrid_bp_nsg_v14_selftest.sbatch
sbatch slurm/fir_hybrid_bp_nsg_v14_smoke.sbatch
sbatch slurm/fir_hybrid_bp_nsg_v14_generate.sbatch
# wait until generation finishes and probe_v14_dataset passes
sbatch slurm/fir_hybrid_bp_nsg_v14_train.sbatch
# wait until training finishes
sbatch slurm/fir_hybrid_bp_nsg_v14_evaluate_report.sbatch
```

The full train job depends on the full generate job. The full evaluation job depends on the full train job. The selftest and smoke jobs write to separate output directories and may be run independently of the full jobs, subject to cluster GPU/CPU resource limits.


Important v14.1 note: smoke runs generation CPU-only inside the GPU allocation, then trains/evaluates with the GPU visible.
