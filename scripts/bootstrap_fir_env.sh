#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$PWD}"
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /cvmfs/soft.computecanada.ca/config/profile/bash.sh /cvmfs/soft.computecanada.ca/config/profile/bash_modules.sh; do
    [ -r "$f" ] && source "$f"
  done
fi
module load python/3.12 2>/dev/null || true
module load scipy-stack 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
if [ ! -d .venv ]; then
  python -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install -e . --no-deps
python scripts/check_runtime_deps.py || true
