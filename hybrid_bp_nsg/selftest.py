from __future__ import annotations

import numpy as np
from typing import Any, Dict

from .bp import bp_decode
from .channels import channel_diagnostics, simulate_frame
from .code import build_code, write_code_summary
from .config import save_resolved_config
from .features import build_rescue_features, build_training_labels


def selftest(cfg: Dict[str, Any]) -> None:
    code = build_code(cfg)
    out_dir = cfg["project"]["output_dir"]
    write_code_summary(code, out_dir)
    save_resolved_config(cfg, out_dir)
    print("Code summary:", code.code_summary())
    for line in channel_diagnostics(code, cfg, cfg.get("eval", {}).get("profiles", ["AWGN"])):
        print(line)
    rng = np.random.default_rng(int(cfg.get("project", {}).get("seed", 123)))
    frame = simulate_frame(code, 6.0, "AWGN", rng, cfg)
    bp = bp_decode(code, frame.llr_internal, iterations=int(cfg.get("bp", {}).get("hybrid_main_iterations", 10)), collect_trace=True)
    print(f"BP selftest success={bp.success} iterations={bp.iterations} syndrome_weight={int(bp.syndrome.sum())}")
    fp = build_rescue_features(code, frame.llr_internal, bp, 6.0, "AWGN", target_basis="bp", num_segments=int(cfg.get("model", {}).get("num_segments", 4)))
    labels = build_training_labels(code, fp, frame.codeword_internal, bp, cfg.get("rescue", {}), cfg.get("model", {}), frame.llr_internal)
    print(f"Feature selftest target_weight={int(labels['target_weight'])} candidate_pos={int((labels['candidate_labels']*labels['candidate_valid']).sum())}")
    print("Selftest complete.")
