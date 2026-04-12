#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p outputs/archive

for src in   outputs/nsgrand_fir_v2_full/repo_export   outputs/nsgrand_fir_v3_full/repo_export
 do
  [ -d "$src" ] || continue
  base="$(basename "$(dirname "$src")")_repo_export"
  dst="outputs/archive/$base"
  if [ ! -d "$dst" ]; then
    cp -a "$src" "$dst"
    echo "Archived $src -> $dst"
  fi
done

rm -f slurm-*.out
rm -rf outputs/nsgrand_fir_v4_smoke outputs/nsgrand_fir_v4_full

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rm --cached --ignore-unmatch slurm-*.out || true
  git rm -r --cached --ignore-unmatch outputs/nsgrand_fir_v4_smoke outputs/nsgrand_fir_v4_full || true
  git rm -r --cached --ignore-unmatch outputs/nsgrand_fir_v3_full/datasets outputs/nsgrand_fir_v3_full/checkpoints outputs/nsgrand_fir_v3_full/logs || true
  echo "Cleaned git index for transient outputs. Review with: git status"
else
  echo "Not inside a git repository; removed transient local files only."
fi
