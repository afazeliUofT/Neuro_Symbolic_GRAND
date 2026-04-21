#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/archive
for d in outputs/hybrid_bp_nsg_v10_selftest outputs/hybrid_bp_nsg_v10_smoke outputs/hybrid_bp_nsg_v10_full outputs/hybrid_bp_nsg_v10_tail; do
  if [ -d "$d/repo_export" ]; then
    ts=$(date +%Y%m%d_%H%M%S)
    cp -a "$d/repo_export" "outputs/archive/$(basename "$d")_repo_export_${ts}" || true
  fi
done
rm -rf outputs/hybrid_bp_nsg_v10_selftest outputs/hybrid_bp_nsg_v10_smoke outputs/hybrid_bp_nsg_v10_full outputs/hybrid_bp_nsg_v10_tail
rm -f slurm-*.out
