#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p outputs/archive

if [ -d outputs/nsgrand_fir_v2_full/repo_export ] && [ ! -d outputs/archive/nsgrand_fir_v2_full_repo_export ]; then
  cp -a outputs/nsgrand_fir_v2_full/repo_export outputs/archive/nsgrand_fir_v2_full_repo_export
  echo "Archived outputs/nsgrand_fir_v2_full/repo_export -> outputs/archive/nsgrand_fir_v2_full_repo_export"
fi

rm -f slurm-*.out
rm -rf outputs/nsgrand_fir_smoke outputs/nsgrand_fir_full outputs/nsgrand_fir_v3_smoke outputs/nsgrand_fir_v3_full

# Remove heavyweight or redundant tracked artifacts from old runs, but keep compact archive bundles.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rm --cached --ignore-unmatch slurm-*.out || true
  git rm -r --cached --ignore-unmatch outputs/nsgrand_fir_smoke outputs/nsgrand_fir_full outputs/nsgrand_fir_v3_smoke outputs/nsgrand_fir_v3_full || true
  git rm -r --cached --ignore-unmatch outputs/nsgrand_fir_v2_full/datasets outputs/nsgrand_fir_v2_full/checkpoints outputs/nsgrand_fir_v2_full/logs || true
  echo "Cleaned git index for transient outputs. Review with: git status"
else
  echo "Not inside a git repository; removed transient local files only."
fi
