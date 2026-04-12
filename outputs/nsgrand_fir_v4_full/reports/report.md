# Neuro-Symbolic GRAND report: nsgrand_fir_v4_full

## Generated artifacts

- `evaluation/evaluation_summary.csv` contains point-level metrics.
- `evaluation/all_raw_records.csv.gz` contains per-sample decoder behavior across all operating points.
- `evaluation/all_paired_records.csv.gz` aligns baseline and nsgrand on the same sample ids.
- `evaluation/strong_vs_nsgrand_paired_records.csv.gz` aligns strong_symbolic and nsgrand on the same sample ids when the strong reference is enabled.
- `reports/overview_by_profile_snr.csv` compiles BLER, query count, latency, fallback, and confidence summaries.
- `reports/paired_summary_by_profile_snr.csv` contains paired-sample baseline-vs-nsgrand win/loss and delta statistics.
- `reports/strong_vs_nsgrand_paired_summary.csv` compares nsgrand against the stronger symbolic reference when available.
- `reports/nsgrand_gate_summary.csv` diagnoses how often each gate and fallback path is used.
- `reports/nsgrand_action_summary.csv` summarizes high-level nsgrand policy actions.
- `reports/interesting_traces_combined.jsonl` stores selected diagnostic traces for hard and representative cases.

## Key findings

- Best non-saturated query reduction occurs at profile=A and snr_db=0.0 with reduction=99.87%
- Largest non-saturated BLER increase for nsgrand relative to baseline is at profile=E and snr_db=6.0, delta=-0.002500
- Largest non-saturated BLER decrease for nsgrand relative to baseline is at profile=E and snr_db=2.0, delta=-0.235000
- Highest paired-sample nsgrand win rate against baseline is at profile=E and snr_db=2.0 with win_rate=0.2417
- Highest paired-sample nsgrand win rate against strong_symbolic is at profile=E and snr_db=2.0 with win_rate=0.2150
- Most common nsgrand policy action is presearch_skip_hopeless with count=13392
- Largest contributor to overall nsgrand query cost is postsearch_skip_after_expanded_fail with avg_queries_contribution=9.630 per sample
- Least efficient post-search fallback regime is search_mode=standard stage=fallback_fail with estimated_success_rate=0.0000
