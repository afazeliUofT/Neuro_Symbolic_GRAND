# Neuro-Symbolic GRAND

This repository is a CPU-oriented research scaffold for an **AI-enhanced GRAND decoder** for finite block-length linear codes, trained on **Sionna-generated 3GPP TR 38.901 TDL channel realizations**. It is designed for Compute Canada / Alliance systems such as **FIR**.

## What is in this v2 package

This package is the **next-step revision** of the original scaffold. It adds three practical upgrades motivated by the first full run:

- **Correct query accounting** for AI-first decoding with fallback.
  - `queries = primary_queries + fallback_queries`
  - latency is split into `primary_elapsed_ms` and `fallback_elapsed_ms`
- **Overflow-aware / early-gated Neuro-Symbolic GRAND**.
  - pre-search fallback on very high predicted overflow or very low confidence
  - expanded AI search when predicted overflow is moderate
  - configurable stronger symbolic fallback after AI exhaustion
- **Comprehensive result export for GitHub-based analysis**.
  - compact `repo_export/` bundle for pushing to GitHub
  - compressed global per-sample records
  - paired baseline-vs-nsgrand records on the same sample ids
  - gate/fallback summaries, calibration summaries, tail-case tables, and selected traces

## Important scope note

This project uses Sionna for **5G NR-standard channel generation** and a **custom finite block-length linear code** for decoder research. It does **not** claim to be a full 5G NR transport-block GRAND decoder.

## Repository layout

- `neuro_symbolic_grand/` — Python package
- `configs/` — FIR configs, including `fir_v2_smoke.yaml` and `fir_v2_full.yaml`
- `scripts/` — convenience launchers and cleanup helper
- `slurm/` — FIR sbatch scripts
- `outputs/` — generated experiment artifacts

## Recommended FIR workflow

After the `.venv` is already installed and verified:

```bash
cd /home/rsadve1/scratch/Neuro_Symbolic_GRAND
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

### Clean the repo for the next iteration

```bash
bash scripts/cleanup_repo_for_git.sh
```

### Optional v2 smoke test

```bash
sbatch slurm/fir_nsgrand_v2_smoke.sbatch
```

### v2 full run

```bash
sbatch slurm/fir_nsgrand_v2_pipeline.sbatch
```

## Outputs from the v2 full run

A full run writes results under:

```text
outputs/nsgrand_fir_v2_full/
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
- `evaluation/nsgrand_gate_summary.csv`
- `reports/overview_by_profile_snr.csv`
- `reports/paired_summary_by_profile_snr.csv`
- `reports/metrics_by_true_error_weight.csv`
- `reports/report.md`

### GitHub-ready compact bundle

The directory below is intentionally designed to be the compact bundle you push to GitHub after the run:

```text
outputs/nsgrand_fir_v2_full/repo_export/
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
