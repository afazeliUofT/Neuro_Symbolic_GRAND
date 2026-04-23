#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_bp_nsg.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Hybrid GRAND YAML config after defaults/normalization")
    ap.add_argument("config", help="YAML config path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg.get("data", {})
    grid = list(data.get("failed_snr_db_grid", []))
    probs = list(data.get("failed_snr_probs", []))
    print(f"Config: {Path(args.config)}")
    print(f"  output_dir: {cfg.get('project', {}).get('output_dir')}")
    print(f"  code: {cfg.get('code', {}).get('family')} k={cfg.get('code', {}).get('k')} n={cfg.get('code', {}).get('n')}")
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
