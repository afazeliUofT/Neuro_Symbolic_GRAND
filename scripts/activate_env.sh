#!/usr/bin/env bash
set -euo pipefail
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -e . --no-deps
