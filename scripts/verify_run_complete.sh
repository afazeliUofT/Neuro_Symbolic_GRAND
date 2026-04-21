#!/bin/bash
set -euo pipefail
ROOT="${1:?need output dir}"
check(){ [ -e "$1" ] || { echo "Missing: $1" >&2; exit 1; }; }
check "$ROOT/artifacts/resolved_config.json"
check "$ROOT/artifacts/runtime_snapshot.json"
check "$ROOT/artifacts/code_summary.json"
check "$ROOT/training/training_summary.json"
check "$ROOT/evaluation/evaluation_summary.csv"
check "$ROOT/reports/report.md"
check "$ROOT/TWC_plots/manifest.csv"
check "$ROOT/TWC_plots/README.md"
