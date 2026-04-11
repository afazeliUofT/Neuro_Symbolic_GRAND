#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/activate_env.sh"
cd "$PROJECT_ROOT"
python -m neuro_symbolic_grand.cli --config configs/fir_full.yaml pipeline
