#!/usr/bin/env bash
set -u
OUT="${1:-outputs/hybrid_bp_nsg_v13_pusch_cdl_full}"
echo "Output dir: $OUT"
echo "--- artifacts ---"
find "$OUT/artifacts" -maxdepth 1 -type f -print 2>/dev/null | sort || true
echo "--- dataset shards ---"
echo "train: $(find "$OUT/datasets/train" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "val:   $(find "$OUT/datasets/val" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "--- checkpoints ---"
ls -lh "$OUT/checkpoints" 2>/dev/null || true
echo "--- training history tail ---"
tail -n 5 "$OUT/training/training_history.csv" 2>/dev/null || true
echo "--- completed evaluation points ---"
find "$OUT/evaluation" -name summary.csv 2>/dev/null | sort || true
echo "--- report ---"
ls -lh "$OUT/reports" 2>/dev/null || true
