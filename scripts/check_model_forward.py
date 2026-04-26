#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from hybrid_bp_nsg.code import build_code
from hybrid_bp_nsg.model import build_rescue_net


def main() -> None:
    ap = argparse.ArgumentParser(description="Check RescueNet forward pass, including mixed precision dtype safety.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    import tensorflow as tf

    mp = cfg.get("train", {}).get("mixed_precision", False)
    if mp:
        policy = str(mp) if isinstance(mp, str) else "mixed_float16"
        if policy == "bfloat16":
            policy = "mixed_bfloat16"
        elif policy == "float16":
            policy = "mixed_float16"
        tf.keras.mixed_precision.set_global_policy(policy)

    code = build_code(cfg)
    b = int(args.batch)
    n = int(code.n)
    m = int(code.m)
    seg = int(cfg.get("model", {}).get("num_segments", 8))
    rsz = int(cfg.get("model", {}).get("rerank_list_size", 16))

    rng = np.random.default_rng(123)
    batch = {
        "var_features": tf.convert_to_tensor(rng.normal(size=(b, n, 16)).astype(np.float16)),
        "check_features": tf.convert_to_tensor(rng.normal(size=(b, m, 6)).astype(np.float16)),
        "global_features": tf.convert_to_tensor(rng.normal(size=(b, 9)).astype(np.float16)),
        "bit_labels": tf.zeros((b, n), dtype=tf.uint8),
        "segment_labels": tf.zeros((b, seg), dtype=tf.uint8),
        "weight_label": tf.zeros((b,), dtype=tf.int16),
        "standard_reachable": tf.zeros((b,), dtype=tf.uint8),
        "expanded_reachable": tf.ones((b,), dtype=tf.uint8),
        "rescueable": tf.ones((b,), dtype=tf.uint8),
        "candidate_features": tf.convert_to_tensor(rng.normal(size=(b, rsz, 14)).astype(np.float16)),
        "candidate_labels": tf.zeros((b, rsz), dtype=tf.uint8),
        "candidate_valid": tf.ones((b, rsz), dtype=tf.uint8),
    }

    model = build_rescue_net(code, cfg)
    out = model(batch, training=False)
    info = {
        "config": args.config,
        "mixed_policy": str(tf.keras.mixed_precision.global_policy()),
        "gpus": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "n": n,
        "m": m,
        "outputs": {k: {"shape": tuple(v.shape.as_list()), "dtype": v.dtype.name} for k, v in out.items()},
    }
    print(json.dumps(info, indent=2, default=str))
    print("Model forward check passed.")


if __name__ == "__main__":
    main()
