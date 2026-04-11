# Neuro-Symbolic GRAND report: nsgrand_fir_smoke

## Generated artifacts

- `evaluation/evaluation_summary.csv` contains point-level metrics.
- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, and fallback rate deltas.
- `evaluation/profile_*/snr_*dB/raw_records.csv` contains per-sample decoder behavior for later forensic analysis.
- `evaluation/profile_*/snr_*dB/traces.jsonl` stores detailed candidate-query traces for sampled blocks.

## Key findings

- Best query reduction occurs at profile=A and snr_db=4.0 with reduction=70.45%
- Largest BLER increase for nsgrand relative to baseline is at profile=A and snr_db=-2.0, delta=-0.004000
