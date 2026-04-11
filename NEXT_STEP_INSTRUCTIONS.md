# Next-step package instructions

1. Unzip this package directly into:

```text
/home/rsadve1/scratch/Neuro_Symbolic_GRAND
```

2. Refresh repo hygiene:

```bash
bash scripts/cleanup_repo_for_git.sh
```

3. Optional quick validation:

```bash
sbatch slurm/fir_nsgrand_v2_smoke.sbatch
```

4. Full run:

```bash
rm -rf outputs/nsgrand_fir_v2_full
sbatch slurm/fir_nsgrand_v2_pipeline.sbatch
```

5. After the full run, the compact GitHub-ready bundle will be available at:

```text
outputs/nsgrand_fir_v2_full/repo_export/
```

6. Push at least that directory, together with the code changes, for later remote analysis.
