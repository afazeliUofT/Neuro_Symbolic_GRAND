# v3 update notes

This revision is specifically motivated by the v2 full-run analysis.

## Why v3 exists

The v2 results showed that:

- the **old query-accounting bug was fixed**
- the **AI stage itself looked useful** at several moderate/high-SNR operating points
- but the **overall hybrid policy was not an efficiency win** because the decoder spent too much time in a very strong direct fallback path on hard cases
- and the **main baseline comparison was scientifically unfair**, because `nsgrand` could inherit gains from a stronger symbolic fallback than the standalone baseline

## Main design changes in v3

1. **Fair fallback for the main comparison**
   - `ns_fallback_kind: baseline` in the recommended FIR configs
   - `nsgrand` is compared to the same symbolic budget/weight envelope as the baseline comparator

2. **Separate strong symbolic reference**
   - `evaluate_strong_symbolic: true`
   - lets you quantify whether `nsgrand` beats the stronger symbolic decoder on matched samples

3. **Hopeless-case skip gate**
   - `overflow_direct_action: skip`
   - intended to reduce wasted work on cases where the model predicts the block is beyond the practical search envelope

4. **Richer exported diagnostics**
   - `policy_action`
   - `strong_vs_nsgrand_paired_records.csv(.gz)`
   - `strong_vs_nsgrand_paired_summary.csv`
   - `nsgrand_action_summary.csv`

## Recommended FIR sequence

```bash
bash scripts/cleanup_before_v3_run.sh
bash scripts/cleanup_repo_for_git.sh
pytest -q
sbatch slurm/fir_nsgrand_v3_smoke.sbatch
# inspect smoke results
sbatch slurm/fir_nsgrand_v3_pipeline.sbatch
```
