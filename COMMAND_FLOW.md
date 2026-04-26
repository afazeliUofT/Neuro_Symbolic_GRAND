# FIR command flow for Hybrid GRAND v14.2

From `~/scratch/Neuro_Symbolic_GRAND` after unzipping this standalone package:

```bash
bash scripts/bootstrap_fir_env.sh "$PWD"
source .venv/bin/activate
python -m pip install -e . --no-deps
python scripts/check_package_version.py
python scripts/check_runtime_deps.py
python scripts/check_config.py configs/fir_hybrid_bp_nsg_full_v14.yaml
python scripts/check_channel_models.py --config configs/fir_hybrid_bp_nsg_full_v14.yaml --profiles AWGN CDL_C
```

Run selftest and smoke first:

```bash
sbatch slurm/fir_hybrid_bp_nsg_v14_selftest.sbatch
sbatch slurm/fir_hybrid_bp_nsg_v14_smoke.sbatch
```

Only after smoke finishes without `Traceback`, `BrokenProcessPool`, or `CUDA_ERROR_NOT_INITIALIZED`, submit full generation:

```bash
sbatch slurm/fir_hybrid_bp_nsg_v14_generate.sbatch
```

Only after the full generation probe reports `target_weight_q1.00 <= 32` and nonzero candidate positives, submit training:

```bash
sbatch slurm/fir_hybrid_bp_nsg_v14_train.sbatch
```

Only after training finishes, submit evaluation/report:

```bash
sbatch slurm/fir_hybrid_bp_nsg_v14_evaluate_report.sbatch
```
