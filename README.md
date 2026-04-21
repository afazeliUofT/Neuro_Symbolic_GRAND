# Hybrid BP + AI-guided GRAND rescue on Sionna 5G NR LDPC — v10 resumable package

This standalone package implements PATH 2:

> A hybrid receiver where NMS/BP is the main LDPC decoder and a graph-aware AI-guided GRAND rescue stage is invoked only on residual LDPC failures.

The package is designed for FIR/Compute Canada runs with a 16-hour wall-time limit. The major change in v10 is **safe resumption**:

- generation resumes at shard granularity and skips already completed `.npz` shards;
- training resumes from `checkpoints/rescue_net_latest.pt` with optimizer and scheduler state;
- evaluation resumes at profile/SNR-point granularity and skips completed `summary.csv` + `raw_records.csv.gz` points;
- reporting can be rerun at any time from completed evaluation outputs;
- the full `pipeline` action is safe to submit repeatedly after wall-time cancellation.

## Important output dirs

```text
outputs/hybrid_bp_nsg_v10_selftest/
outputs/hybrid_bp_nsg_v10_smoke/
outputs/hybrid_bp_nsg_v10_full/
outputs/hybrid_bp_nsg_v10_tail/
```

## Run order

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH}"
pytest -q
sbatch slurm/fir_hybrid_bp_nsg_selftest.sbatch
```

After selftest succeeds, run either the all-in-one resumable pipeline repeatedly:

```bash
sbatch slurm/fir_hybrid_bp_nsg_pipeline.sbatch
```

or run explicit stages:

```bash
sbatch slurm/fir_hybrid_bp_nsg_generate.sbatch
sbatch slurm/fir_hybrid_bp_nsg_train.sbatch
sbatch slurm/fir_hybrid_bp_nsg_evaluate_report.sbatch
```

After full evaluation/report completes:

```bash
sbatch slurm/fir_hybrid_bp_nsg_tail_eval.sbatch
```

## Checking status

```bash
bash scripts/status.sh outputs/hybrid_bp_nsg_v10_full
```

This reports shard counts, partial training history, completed evaluation points, and whether reports/TWC plots exist.

## Why generation can be long

The full training set is built from **failed NMS-20 states**. Since BP/NMS succeeds often, especially at high SNR, the generator may simulate many more packets than the number of kept training examples. v10 therefore uses smaller shards and a failure-biased training SNR distribution so progress is saved frequently.

## Publication plots

When the full report stage completes, it creates:

```text
outputs/hybrid_bp_nsg_v10_full/TWC_plots/manifest.csv
outputs/hybrid_bp_nsg_v10_full/TWC_plots/README.md
outputs/hybrid_bp_nsg_v10_full/TWC_plots/*.png
outputs/hybrid_bp_nsg_v10_full/TWC_plots/*.pdf
outputs/hybrid_bp_nsg_v10_full/TWC_plots/*.csv
```

`README.md` embeds the PNGs for direct GitHub viewing. All BLER curves use log scale.

## Current hybrid iteration structure

- Hybrid main LDPC stage: NMS, max 20 iterations
- Strong reference baseline: NMS, max 50 iterations
- Hybrid micro-BP refinement: NMS, max 8 iterations on a tiny AI-ranked shortlist
- Early syndrome stopping is enabled.

## Monte Carlo policy

The evaluation runs sequential Monte Carlo per profile/SNR point:

- stop when target frame errors are reached for target decoders, or when the sample cap is reached;
- mark a BLER point as plot-eligible only if it has at least the configured minimum number of frame errors.

## Notes on standard Sionna NR LDPC

The code path uses Sionna's built-in NR LDPC encoder path and the corresponding internal graph representation. The actual instantiated code/graph dimensions are written to:

```text
outputs/.../artifacts/code_summary.json
```
