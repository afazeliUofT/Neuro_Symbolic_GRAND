#!/usr/bin/env bash
set -u
OUT="${1:-outputs/hybrid_bp_nsg_v13_pusch_cdl_full}"
test -f "$OUT/checkpoints/rescue_net_tf.best.weights.h5" && echo "best checkpoint: OK" || echo "best checkpoint: missing"
test -f "$OUT/checkpoints/rescue_net_tf.weights.h5" && echo "final checkpoint: OK" || echo "final checkpoint: missing"
test -f "$OUT/evaluation/evaluation_summary.csv" && echo "evaluation summary: OK" || echo "evaluation summary: missing"
