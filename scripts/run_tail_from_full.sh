#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_DIR="${1:-$ROOT/outputs/hybrid_bp_nsg_v10_full}"
TAIL_DIR="${2:-$ROOT/outputs/hybrid_bp_nsg_v10_tail}"
CFG_FULL="${3:-$ROOT/configs/fir_hybrid_bp_nsg_full.yaml}"
CFG_TAIL="${4:-$ROOT/configs/fir_hybrid_bp_nsg_tail.yaml}"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TAIL_DIR/checkpoints" "$TAIL_DIR/artifacts" "$TAIL_DIR/training"
[ -f "$FULL_DIR/checkpoints/rescue_net.pt" ] || { echo "Missing checkpoint: $FULL_DIR/checkpoints/rescue_net.pt" >&2; exit 1; }
cp "$FULL_DIR/checkpoints/rescue_net.pt" "$TAIL_DIR/checkpoints/rescue_net.pt"
cp "$FULL_DIR/artifacts/resolved_config.json" "$TAIL_DIR/artifacts/" || true
cp "$FULL_DIR/artifacts/runtime_snapshot.json" "$TAIL_DIR/artifacts/" || true
cp "$FULL_DIR/artifacts/code_summary.json" "$TAIL_DIR/artifacts/" || true
cp "$FULL_DIR/training/training_summary.json" "$TAIL_DIR/training/" || true
python -m hybrid_bp_nsg.cli --config "$CFG_TAIL" tail_evaluate
python -m hybrid_bp_nsg.cli --config "$CFG_TAIL" report
python -m hybrid_bp_nsg.cli --config "$CFG_FULL" report --tail-summary-path "$TAIL_DIR/evaluation/evaluation_summary.csv"

"$ROOT/scripts/verify_run_complete.sh" "$TAIL_DIR"
"$ROOT/scripts/verify_run_complete.sh" "$FULL_DIR"
