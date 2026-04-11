# Neuro-Symbolic GRAND report: nsgrand_fir_full

## Generated artifacts

- `evaluation/evaluation_summary.csv` contains point-level metrics.
- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, and fallback rate deltas.
- `evaluation/profile_*/snr_*dB/raw_records.csv` contains per-sample decoder behavior for later forensic analysis.
- `evaluation/profile_*/snr_*dB/traces.jsonl` stores detailed candidate-query traces for sampled blocks.

## Key findings

- Best query reduction occurs at profile=E and snr_db=4.0 with reduction=67.54%
- Largest BLER increase for nsgrand relative to baseline is at profile=E and snr_db=6.0, delta=0.000833
