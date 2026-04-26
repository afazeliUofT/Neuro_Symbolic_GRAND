#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# When this script is run manually on the FIR login node there is no GPU, but the
# TensorFlow build is GPU-enabled. Hide GPUs there to avoid noisy cuInit warnings.
if not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_bp_nsg.config import load_config
from hybrid_bp_nsg.codes.factory import build_code
from hybrid_bp_nsg.channels.simulator import simulate_frame
from hybrid_bp_nsg.decoders.bp import BeliefPropagationDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-check AWGN/CDL_C channel simulation and LLR interface")
    ap.add_argument("--config", default="configs/fir_hybrid_bp_nsg_smoke.yaml")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--snr-db", type=float, default=2.0)
    ap.add_argument("--decode-smoke", action="store_true", help="Also run the BP decoder and print its status. Not required for channel validation.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    profiles = args.profiles or list(cfg.get("eval", {}).get("profiles", ["AWGN", "CDL_C"]))
    code = build_code(cfg["code"])
    bp = BeliefPropagationDecoder(code, max_iters=int(cfg["bp"].get("hybrid_main_iterations", 20))) if args.decode_smoke else None
    rng = np.random.default_rng(int(cfg["project"].get("seed", 1234)) + 777)

    print("code_family:", code.family)
    print("transport_k:", code.transport_k)
    print("ldpc_k:", code.k)
    print("n_internal:", code.n)
    print("n_transmitted:", code.transmitted_n)
    print("profiles:", profiles)
    for prof in profiles:
        frame = simulate_frame(code, float(args.snr_db), str(prof), rng, channel_cfg=cfg.get("channel", {}))
        finite_ok = bool(np.isfinite(frame.llr_tx).all() and np.isfinite(frame.llr_internal).all())
        msg = (
            f"profile={prof} tx_len={frame.codeword_tx.size} llr_tx_len={frame.llr_tx.size} "
            f"llr_internal_len={frame.llr_internal.size} channel_type={frame.channel_meta.get('channel_type')} "
            f"modulation={frame.channel_meta.get('modulation')} perfect_csi={frame.channel_meta.get('perfect_csi')} "
            f"equalizer={frame.channel_meta.get('equalizer')} finite_llr={finite_ok}"
        )
        if bp is not None:
            res = bp.decode(frame.llr_internal, collect_trace=False)
            msg += f" bp_success={res.success} crc_ok={res.crc_ok} sw={int(res.syndrome.sum())}"
        print(msg)
        if frame.codeword_tx.size != code.transmitted_n:
            raise SystemExit(f"ERROR: transmitted length mismatch for profile {prof}")
        if frame.llr_tx.size != code.transmitted_n:
            raise SystemExit(f"ERROR: transmitted-LLR length mismatch for profile {prof}")
        if frame.llr_internal.size != code.n:
            raise SystemExit(f"ERROR: internal-LLR length mismatch for profile {prof}")
        if not finite_ok:
            raise SystemExit(f"ERROR: non-finite LLR values for profile {prof}")


if __name__ == "__main__":
    main()
