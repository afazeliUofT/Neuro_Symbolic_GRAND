#!/bin/bash
set -euo pipefail
ROOT="${1:-outputs/hybrid_bp_nsg_v10_full}"
echo "Status for $ROOT"
echo "Generated train shards: $(find "$ROOT/datasets/train" -name 'train_shard_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Generated val shards:   $(find "$ROOT/datasets/val" -name 'val_shard_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
[ -f "$ROOT/training/training_history_partial.csv" ] && tail -n 5 "$ROOT/training/training_history_partial.csv" || true
[ -f "$ROOT/training/training_summary.json" ] && cat "$ROOT/training/training_summary.json" || true
echo "Completed eval points: $(find "$ROOT/evaluation" -name summary.csv 2>/dev/null | wc -l | tr -d ' ')"
[ -f "$ROOT/reports/report.md" ] && echo "Report exists: $ROOT/reports/report.md" || true
[ -f "$ROOT/TWC_plots/manifest.csv" ] && echo "TWC plots manifest exists: $ROOT/TWC_plots/manifest.csv" || true
