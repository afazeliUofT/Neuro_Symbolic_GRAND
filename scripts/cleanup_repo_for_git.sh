#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

cat > .gitignore <<'EOF'
# Virtual environments and caches
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.pip_cache/
.probe_env_*/
probe_logs/
probe_logs_*.tar.gz

# Slurm and runtime logs
slurm-*.out
*.log

# Packaging archives
*.tar.gz
*.zip

# Generated outputs to keep out of Git by default
outputs/**/datasets/
outputs/**/checkpoints/
outputs/**/logs/

# Keep compact analysis bundles and reports trackable
!outputs/**/repo_export/
!outputs/**/repo_export/**
!outputs/**/reports/
!outputs/**/reports/**
!outputs/**/evaluation/
!outputs/**/evaluation/**
!outputs/**/training/training_history.csv
!outputs/**/training/training_summary.json
!outputs/**/artifacts/resolved_config.json
!outputs/**/artifacts/runtime_snapshot.json
EOF

echo "Wrote refreshed .gitignore"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for f in slurm-*.out; do
    [ -e "$f" ] || continue
    git rm --cached --ignore-unmatch "$f" || true
  done
  for d in outputs/*/datasets outputs/*/checkpoints outputs/*/logs; do
    [ -e "$d" ] || continue
    git rm -r --cached --ignore-unmatch "$d" || true
  done
  echo "Git index cleaned for transient artifacts. Review with: git status"
else
  echo "Not inside a git repository; only .gitignore was refreshed."
fi
