#!/usr/bin/env bash
set -u
OUT="${1:-outputs/hybrid_bp_nsg_v11_channel_aligned_full}"
test -f "$OUT/checkpoints/rescue_net.pt" && echo "checkpoint: OK" || echo "checkpoint: missing"
test -f "$OUT/evaluation/evaluation_summary.csv" && echo "evaluation summary: OK" || echo "evaluation summary: missing"
