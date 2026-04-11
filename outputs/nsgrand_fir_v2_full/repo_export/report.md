# Neuro-Symbolic GRAND report: nsgrand_fir_v2_full

## Generated artifacts

- `evaluation/evaluation_summary.csv` contains point-level metrics.
- `evaluation/all_raw_records.csv.gz` contains per-sample decoder behavior across all operating points.
- `evaluation/all_paired_records.csv.gz` aligns baseline and nsgrand on the same sample ids.
- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, fallback, and confidence summaries.
- `reports/paired_summary_by_profile_snr.csv` contains paired-sample win/loss and delta statistics.
- `reports/nsgrand_gate_summary.csv` diagnoses how often each gate and fallback path is used.
- `reports/interesting_traces_combined.jsonl` stores selected diagnostic traces for hard and representative cases.

## Key findings

- Best query reduction occurs at profile=E and snr_db=6.0 with reduction=45.19%
- Largest BLER increase for nsgrand relative to baseline is at profile=A and snr_db=-4.0, delta=0.000000
- Largest BLER decrease for nsgrand relative to baseline is at profile=E and snr_db=2.0, delta=-0.155000
- Highest paired-sample nsgrand win rate is at profile=E and snr_db=2.0 with win_rate=0.1583
