# Next-step instructions for v4

```bash
cd /home/rsadve1/scratch/Neuro_Symbolic_GRAND
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

## Clean before the next run

```bash
bash scripts/cleanup_before_v4_run.sh
bash scripts/cleanup_repo_for_git.sh
```

## Quick checks

```bash
pytest -q
```

## Optional micro-smoke

```bash
sbatch slurm/fir_nsgrand_v4_smoke.sbatch
```

## Full v4 run

```bash
sbatch slurm/fir_nsgrand_v4_pipeline.sbatch
```

## Files to inspect or push after the run

```text
outputs/nsgrand_fir_v4_full/repo_export/report.md
outputs/nsgrand_fir_v4_full/repo_export/overview_by_profile_snr.csv
outputs/nsgrand_fir_v4_full/repo_export/overview_nontrivial_by_profile_snr.csv
outputs/nsgrand_fir_v4_full/repo_export/nsgrand_action_contribution_summary.csv
outputs/nsgrand_fir_v4_full/repo_export/postsearch_outcome_summary.csv
outputs/nsgrand_fir_v4_full/repo_export/all_raw_records.csv.gz
outputs/nsgrand_fir_v4_full/repo_export/all_paired_records.csv.gz
```
