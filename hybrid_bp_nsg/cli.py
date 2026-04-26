from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .utils.env import dependency_report
from .utils.io import ensure_dir, write_json
from .utils.logging import get_logger


def action_selftest(cfg):
    from .codes.factory import build_code
    from .channels.simulator import simulate_frame
    from .decoders.bp import BeliefPropagationDecoder
    from .decoders.hybrid import HybridBPNSGDecoder

    logger = get_logger("selftest")
    print(dependency_report())
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    ensure_dir(out_dir / "artifacts")
    write_json(cfg, out_dir / "artifacts" / "resolved_config.json")

    code = build_code(cfg["code"])
    summary = {
        "family": code.family,
        "n_internal": code.n,
        "n_transmitted": code.transmitted_n,
        "k": code.k,
        "transport_k": int(getattr(code, "transport_k", code.k)),
        "m": code.m,
        "edges": int(code.h.sum()),
        "punctured": int(code.punctured_positions.size),
        "metadata": code.metadata,
    }
    write_json(summary, out_dir / "artifacts" / "code_summary.json")
    logger.info("Code summary: %s", summary)

    rng = np.random.default_rng(int(cfg["project"].get("seed", 123)))
    for i in range(5):
        msg = rng.integers(0, 2, size=int(getattr(code, "transport_k", code.k)), dtype=np.uint8)
        c, tx = code.encode_payload(msg)
        sw = int(code.syndrome(c).sum())
        if sw != 0:
            raise RuntimeError(f"encoder/internal PCM validation failed in selftest at sample {i}: syndrome weight {sw}")
        if tx.shape[-1] != code.transmitted_n:
            raise RuntimeError(f"transmitted length mismatch in selftest: got {tx.shape[-1]} expected {code.transmitted_n}")
        if getattr(code, "has_outer_crc", False) and not code.crc_check_internal(c):
            raise RuntimeError(f"CRC validation failed in selftest at sample {i}")

    bp = BeliefPropagationDecoder(code, max_iters=int(cfg["bp"]["hybrid_main_iterations"]), nms_alpha=float(cfg["bp"]["nms_alpha"]))
    frame = simulate_frame(code, snr_db=2.0, profile=str(cfg.get("eval", {}).get("profiles", ["AWGN"])[0]), rng=rng, channel_cfg=cfg.get("channel", {}))
    r = bp.decode(frame.llr_internal, collect_trace=True)
    logger.info("BP selftest success=%s iterations=%d syndrome_weight=%d", r.success, r.iterations_used, int(r.syndrome.sum()))

    # Heuristic hybrid smoke decode without requiring a checkpoint.
    cfg2 = dict(cfg)
    cfg2["rescue"] = dict(cfg["rescue"])
    cfg2["rescue"]["mode"] = "orb"
    hyb = HybridBPNSGDecoder(code, cfg2, rescue_net=None, mode="orb")
    hr = hyb.decode(frame.llr_internal, snr_db=2.0, profile=str(cfg.get("eval", {}).get("profiles", ["AWGN"])[0]), collect_trace=True)
    logger.info("Hybrid selftest success=%s action=%s queries=%d", hr.success, hr.action, hr.queries)
    logger.info("Selftest complete.")


def main():
    parser = argparse.ArgumentParser(description="Hybrid BP + Channel-Aligned AI/Tanner-GRAND rescue")
    parser.add_argument("action", choices=["selftest", "generate", "train", "evaluate", "report", "pipeline", "plots"])
    parser.add_argument("--config", "-c", default="configs/fir_hybrid_bp_nsg_selftest.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.action == "selftest":
        action_selftest(cfg)
    elif args.action == "generate":
        from .training.generation import generate_dataset
        generate_dataset(cfg)
    elif args.action == "train":
        from .training.train import train_model
        train_model(cfg)
    elif args.action == "evaluate":
        from .training.evaluation import evaluate
        evaluate(cfg)
    elif args.action == "report":
        from .analysis.reporting import make_report
        make_report(cfg)
    elif args.action == "plots":
        from .plotting.twc_plots import make_plots
        make_plots(cfg)
    elif args.action == "pipeline":
        from .training.generation import generate_dataset
        generate_dataset(cfg)
        from .training.train import train_model
        train_model(cfg)
        from .training.evaluation import evaluate
        evaluate(cfg)
        from .analysis.reporting import make_report
        make_report(cfg)


if __name__ == "__main__":
    main()
