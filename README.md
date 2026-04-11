# Neuro-Symbolic GRAND

This repository is a Python research scaffold for a **Neuro-Symbolic GRAND** decoder for finite block-length linear codes, trained on **Sionna-generated 3GPP TR 38.901 TDL channel realizations**. The package is designed for CPU-only execution on Compute Canada / Alliance systems such as FIR.

## What is implemented

- A finite block-length **systematic sparse linear code** builder.
- A **weighted GRAND-like baseline** that searches reliability-ordered error patterns.
- A **Neuro-Symbolic GRAND** decoder that uses a learned posterior scheduler to prioritize GRAND candidate branches and falls back to a strong symbolic baseline when confidence is low.
- **Sionna-based TDL channel generation** for training and evaluation, with a clearly marked fallback stochastic channel if Sionna execution fails at runtime.
- Parallel **train/validation data generation**, model training, Monte Carlo evaluation, and report generation.
- Detailed output artifacts for later bottleneck analysis:
  - point-level BLER/BER/query/latency summaries
  - per-sample raw decoder records
  - sampled query traces
  - training curves and comparative plots

## Important scope note

The current package uses Sionna for **5G NR-standard channel generation** and a custom finite block-length code for decoder research. It does **not** claim to be a full 5G NR transport-block GRAND decoder.

## Repository layout

- `neuro_symbolic_grand/` — Python package
- `configs/` — smoke and full FIR configs
- `scripts/` — convenience launchers using the `.venv`
- `slurm/` — FIR sbatch scripts
- `outputs/` — generated experiment artifacts

## Expected workflow on FIR

After the `.venv` is installed and verified:

```bash
cd /home/rsadve1/scratch/Neuro_Symbolic_GRAND
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

### Smoke test

```bash
sbatch slurm/fir_nsgrand_smoke.sbatch
```

### Full run

```bash
sbatch slurm/fir_nsgrand_pipeline.sbatch
```

## Outputs

A full run writes results under the configured output directory, e.g.

```text
outputs/nsgrand_fir_full/
  artifacts/
  checkpoints/
  datasets/
  training/
  evaluation/
  reports/
  logs/
```

The most useful files for later analysis are:

- `evaluation/evaluation_summary.csv`
- `reports/overview_by_profile_snr.csv`
- `evaluation/profile_*/snr_*dB/raw_records.csv`
- `evaluation/profile_*/snr_*dB/traces.jsonl`
- `reports/report.md`

## Packaging note

The Slurm scripts assume the project is unpacked directly in:

```text
/home/rsadve1/scratch/Neuro_Symbolic_GRAND
```

and that the virtual environment lives at:

```text
/home/rsadve1/scratch/Neuro_Symbolic_GRAND/.venv
```
