#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-outputs/hybrid_bp_nsg_v11_channel_aligned_full}"
rm -rf "$OUT"
echo "Removed $OUT"
