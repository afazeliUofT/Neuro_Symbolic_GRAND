# Neuro-Symbolic GRAND

This repository is a CPU-oriented research scaffold for an **AI-enhanced GRAND decoder** for finite block-length linear codes, trained on **Sionna-generated 3GPP TR 38.901 TDL channel realizations**. It is designed for Compute Canada / Alliance systems such as **FIR**.

## What is in this v3 package

This package is the **fairness-and-efficiency revision** after analyzing the v2 full run.

It adds four targeted upgrades:

- **Fairer scientific comparison**.
  - the main `nsgrand` path can now use a fallback that is **equalized to the baseline**
  - a separate `strong_symbolic` reference can be evaluated in parallel
  - this separates “AI scheduling gain” from “stronger symbolic fallback gain”

- **Hopeless-case skipping instead of expensive direct fallback**.
  - a new `overflow_direct_action` policy supports `skip`, `fallback`, or `disabled`
  - the recommended v3 FIR configs use `skip` for extreme predicted-overflow cases
  - this is meant to cut wasted computation at very low SNR / very high true error weight

- **Expanded diagnostics for policy analysis**.
  - `policy_action` is logged per sample
  - `nsgrand_action_summary.csv` is exported per point and globally
  - `strong_vs_nsgrand_paired_records.csv(.gz)` and a paired summary are exported when the strong reference is enabled

- **GitHub-ready export remains compact**.
  - `repo_export/` still contains the tables, plots, traces, training summaries, and compressed per-sample records needed for later analysis

## Important scope note

This project uses Sionna for **5G NR-standard channel generation** and a **custom finite block-length linear code** for decoder research. It does **not** claim to be a full 5G NR transport-block GRAND decoder.

## Repository layout

- `neuro_symbolic_grand/` — Python package
- `configs/` — FIR configs, including `fir_v3_smoke.yaml` and `fir_v3_full.yaml`
- `scripts/` — convenience launchers and cleanup helpers
- `slurm/` — FIR sbatch scripts
- `outputs/` — generated experiment artifacts

## Recommended FIR workflow

After the `.venv` is already installed and verified:

```bash
cd /home/rsadve1/scratch/Neuro_Symbolic_GRAND
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

### Clean the repo and old outputs before v3

```bash
bash scripts/cleanup_before_v3_run.sh
bash scripts/cleanup_repo_for_git.sh
```

### Optional v3 smoke test

```bash
sbatch slurm/fir_nsgrand_v3_smoke.sbatch
```

### v3 full run

```bash
sbatch slurm/fir_nsgrand_v3_pipeline.sbatch
```

## Outputs from the v3 full run

A full run writes results under:

```text
outputs/nsgrand_fir_v3_full/
  artifacts/
  checkpoints/
  datasets/
  training/
  evaluation/
  reports/
  repo_export/
  logs/
```

### Most useful local analysis files

- `evaluation/evaluation_summary.csv`
- `evaluation/all_raw_records.csv.gz`
- `evaluation/all_paired_records.csv.gz`
- `evaluation/strong_vs_nsgrand_paired_records.csv.gz`
- `evaluation/nsgrand_gate_summary.csv`
- `evaluation/nsgrand_action_summary.csv`
- `reports/overview_by_profile_snr.csv`
- `reports/paired_summary_by_profile_snr.csv`
- `reports/strong_vs_nsgrand_paired_summary.csv`
- `reports/metrics_by_true_error_weight.csv`
- `reports/report.md`

### GitHub-ready compact bundle

The directory below is intentionally designed to be the compact bundle you push to GitHub after the run:

```text
outputs/nsgrand_fir_v3_full/repo_export/
```

It keeps:

- point-level and paired summaries
- compressed per-sample records
- selected diagnostic traces
- training summaries
- plots
- resolved config and runtime snapshot

while excluding the heavyweight dataset shards, checkpoints, and runtime logs.

## Packaging note

The Slurm scripts assume the project is unpacked directly in:

```text
/home/rsadve1/scratch/Neuro_Symbolic_GRAND
```

and that the virtual environment lives at:

```text
/home/rsadve1/scratch/Neuro_Symbolic_GRAND/.venv
```
