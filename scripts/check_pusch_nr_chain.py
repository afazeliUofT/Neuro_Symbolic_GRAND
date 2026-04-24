#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_bp_nsg.config import load_config
from hybrid_bp_nsg.codes.factory import build_code


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the Sionna NR PUSCH-derived LDPC chain")
    ap.add_argument("--config", default="configs/fir_hybrid_bp_nsg_full.yaml")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    code = build_code(cfg["code"])
    print("family:", code.family)
    print("transport_k:", code.transport_k)
    print("ldpc_k:", code.k)
    print("n_internal:", code.n)
    print("n_transmitted:", code.transmitted_n)
    print("rate:", code.rate)
    print("metadata:", code.metadata)

    rng = np.random.default_rng(int(cfg["project"].get("seed", 1234)) + 99)
    for t in range(int(args.trials)):
        payload = rng.integers(0, 2, size=code.transport_k, dtype=np.uint8)
        internal, tx = code.encode_payload(payload)
        sw = int(code.syndrome(internal).sum())
        crc_ok = bool(code.crc_check_internal(internal))
        recovered = code.payload_bits(internal)
        tx_match = None
        if hasattr(code, "tb_encoder"):
            try:
                sionna_tx = code.tb_encoder(payload[None, :].astype(np.float32))
                sionna_tx = np.asarray(sionna_tx.numpy() if hasattr(sionna_tx, "numpy") else sionna_tx)
                sionna_tx = (sionna_tx.reshape(-1) > 0.5).astype(np.uint8)
                tx_match = bool(np.array_equal(tx.reshape(-1), sionna_tx.reshape(-1)))
            except Exception as e:
                tx_match = f"probe_failed:{type(e).__name__}:{e}"
        print(f"trial={t} syndrome_weight={sw} crc_ok={crc_ok} payload_match={np.array_equal(payload, recovered)} tx_match_sionna={tx_match} tx_len={tx.shape[-1]}")
        if sw != 0:
            raise SystemExit("ERROR: nonzero syndrome on encoded codeword")
        if code.has_outer_crc and not crc_ok:
            raise SystemExit("ERROR: CRC failed on encoder output")
        if tx_match is False:
            raise SystemExit("ERROR: internal_to_tx mapping does not match Sionna TBEncoder output")


if __name__ == "__main__":
    main()
