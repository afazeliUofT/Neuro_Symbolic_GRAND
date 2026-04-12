# Neuro-Symbolic GRAND

This repository is a CPU-oriented research scaffold for an **AI-enhanced GRAND decoder** for finite block-length linear codes, trained on **Sionna-generated 3GPP TR 38.901 TDL channel realizations**. It is designed for Compute Canada / Alliance systems such as **FIR**.

## What is in this v4 package

This package is the **post-v3 efficiency-and-analysis revision**.

It focuses on the main finding from the v3 full run: most remaining query cost came from **post-search fallback after expanded-overflow AI search failures**, while that branch had a very low fallback success rate. The v4 package therefore adds four targeted upgrades:

- **Mode-specific post-search fallback control**.
  - `fallback_after_standard_ai_fail`
  - `fallback_after_expanded_ai_fail`
  - the recommended v4 config keeps fallback after standard AI failures, but **skips fallback after expanded-overflow AI failures**

- **Cleaner diagnostics for where compute is going**.
  - `nsgrand_action_contribution_summary.csv`
  - `postsearch_outcome_summary.csv`
  - these quantify which policy actions dominate total query count, latency, and errors

- **Better headline reporting**.
  - `overview_nontrivial_by_profile_snr.csv` excludes saturated baseline points where BLER is already essentially 1.0
  - the markdown report now highlights **non-saturated** findings by default

- **A real micro-smoke configuration**.
  - `fir_v4_smoke.yaml` is intentionally much smaller than previous smoke jobs
  - it is designed as a quick preflight, not a mini full run

## Important scope note

This project uses Sionna for **5G NR-standard channel generation** and a **custom finite block-length linear code** for decoder research. It does **not** claim to be a full 5G NR transport-block GRAND decoder.

## Repository layout

- `neuro_symbolic_grand/` — Python package
- `configs/` — FIR configs, including `fir_v4_smoke.yaml` and `fir_v4_full.yaml`
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

### Clean the repo and old outputs before v4

```bash
bash scripts/cleanup_before_v4_run.sh
bash scripts/cleanup_repo_for_git.sh
```

### Optional v4 micro-smoke

```bash
sbatch slurm/fir_nsgrand_v4_smoke.sbatch
```

### v4 full run

```bash
sbatch slurm/fir_nsgrand_v4_pipeline.sbatch
```

## Outputs from the v4 full run

A full run writes results under:

```text
outputs/nsgrand_fir_v4_full/
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
- `reports/overview_nontrivial_by_profile_snr.csv`
- `reports/nsgrand_action_contribution_summary.csv`
- `reports/postsearch_outcome_summary.csv`
- `reports/report.md`

### GitHub-ready compact bundle

The directory below is intentionally designed to be the compact bundle you push to GitHub after the run:

```text
outputs/nsgrand_fir_v4_full/repo_export/
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
