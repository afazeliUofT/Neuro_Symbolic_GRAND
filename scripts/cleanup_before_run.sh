#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-outputs/hybrid_bp_nsg_v13_pusch_cdl_full}"
rm -rf "$OUT"
echo "Removed $OUT"
