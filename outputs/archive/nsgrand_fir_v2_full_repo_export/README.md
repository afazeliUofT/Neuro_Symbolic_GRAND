# GitHub-ready export bundle: nsgrand_fir_v2_full

This directory is the compact bundle intended to be pushed to GitHub for later remote analysis.
It excludes heavyweight dataset shards, checkpoints, and runtime logs, but retains the key aggregated tables,
compressed per-sample records, selected traces, training summaries, and plots.

Most useful files:
- `evaluation_summary.csv`
- `overview_by_profile_snr.csv`
- `paired_summary_by_profile_snr.csv`
- `all_raw_records.csv.gz`
- `all_paired_records.csv.gz`
- `nsgrand_gate_summary.csv`
- `interesting_traces_combined.jsonl`
- `report.md`
