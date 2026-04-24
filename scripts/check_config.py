#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_bp_nsg.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Hybrid GRAND YAML config after defaults/normalization")
    ap.add_argument("config", help="YAML config path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg.get("data", {})
    code = cfg.get("code", {})
    grid = list(data.get("failed_snr_db_grid", []))
    probs = list(data.get("failed_snr_probs", []))
    code_n = code.get("n", code.get("num_coded_bits"))
    code_k = code.get("k", code.get("target_tb_size", code.get("payload_k")))
    print(f"Config: {Path(args.config)}")
    print(f"  output_dir: {cfg.get('project', {}).get('output_dir')}")
    print(f"  code: {code.get('family')} k={code_k} n={code_n}")
    print(f"  failed_snr_db_grid: {grid}")
    print(f"  failed_snr_probs: {probs}")
    print(f"  failed_snr_probs_note: {data.get('failed_snr_probs_note', 'none')}")
    print(f"  probability_sum: {sum(float(x) for x in probs):.12f}")
    if len(grid) != len(probs):
        raise SystemExit(f"ERROR: len(failed_snr_db_grid)={len(grid)} != len(failed_snr_probs)={len(probs)}")
    if not grid:
        raise SystemExit("ERROR: failed_snr_db_grid is empty")
    print("Config validation OK")


if __name__ == "__main__":
    main()
