from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..channels.simulator import simulate_frame
from ..codes.factory import build_code
from ..decoders.bp import BeliefPropagationDecoder
from ..decoders.hybrid import HybridBPNSGDecoder
from ..models.rescue_net import RescueNet
from ..training.dataset import infer_shapes
from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logging import get_logger


def _load_rescue_net(cfg: Dict[str, object], code, device: str = "cpu"):
    try:
        import torch
    except Exception:
        return None
    ckpt_path = Path(cfg["project"]["output_dir"]) / "checkpoints" / "rescue_net.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    shapes = ckpt.get("shapes")
    if shapes is None:
        try:
            shapes = infer_shapes(Path(cfg["project"]["output_dir"]) / "datasets" / "train")
        except Exception:
            shapes = {
                "num_var_features": 16,
                "num_check_features": 6,
                "num_global_features": 9,
                "n": code.n,
                "m": code.m,
                "candidate_feature_dim": 8,
            }
    model = RescueNet(
        num_var_features=int(shapes["num_var_features"]),
        num_check_features=int(shapes["num_check_features"]),
        num_global_features=int(shapes["num_global_features"]),
        n=code.n,
        m=code.m,
        num_segments=int(cfg["model"]["num_segments"]),
        max_weight_class=int(cfg["model"]["max_weight_class"]),
        hidden_dim=int(cfg["model"]["graph_hidden_dim"]),
        graph_layers=int(cfg["model"]["graph_layers"]),
        top_k_tokens=int(cfg["model"]["top_k_tokens"]),
        transformer_heads=int(cfg["model"].get("transformer_heads", 4)),
        transformer_layers=int(cfg["model"].get("transformer_layers", 1)),
        dropout=float(cfg["train"].get("dropout", 0.05)),
        candidate_feature_dim=int(shapes.get("candidate_feature_dim", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def evaluate(cfg: Dict[str, object]) -> None:
    logger = get_logger("evaluate")
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    eval_root = ensure_dir(out_dir / "evaluation")
    write_json(cfg, out_dir / "artifacts" / "resolved_config.json")
    code = build_code(cfg["code"])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rescue_net = _load_rescue_net(cfg, code, device=device)
    if rescue_net is None:
        logger.info("No trained rescue_net.pt found; hybrid uses heuristic channel-aligned Tanner-GRAND policy.")
    else:
        logger.info("Loaded rescue network on %s", device)

    bp20 = BeliefPropagationDecoder(
        code,
        max_iters=int(cfg["bp"].get("hybrid_main_iterations", 20)),
        algorithm=str(cfg["bp"].get("hybrid_main_algorithm", "nms")),
        nms_alpha=float(cfg["bp"].get("nms_alpha", 0.8)),
        early_stop=bool(cfg["bp"].get("early_stop", True)),
    )
    bp50 = BeliefPropagationDecoder(
        code,
        max_iters=int(cfg["bp"].get("strong_iterations", 50)),
        algorithm=str(cfg["bp"].get("hybrid_main_algorithm", "nms")),
        nms_alpha=float(cfg["bp"].get("nms_alpha", 0.8)),
        early_stop=bool(cfg["bp"].get("early_stop", True)),
    )
    hybrid = HybridBPNSGDecoder(code, cfg, rescue_net=rescue_net, device=device, mode=str(cfg["rescue"].get("mode", "ai")))

    rng = np.random.default_rng(int(cfg["project"].get("seed", 1234)) + 999)
    all_summaries = []
    profiles = list(cfg["eval"].get("profiles", ["A"]))
    snrs = list(cfg["eval"].get("snr_db_grid", [0]))
    for profile in profiles:
        for snr in snrs:
            point_dir = ensure_dir(eval_root / f"profile_{profile}" / f"snr_{float(snr):+0.1f}dB")
            summary_path = point_dir / "summary.csv"
            raw_path = point_dir / "raw_records.csv.gz"
            if summary_path.exists() and raw_path.exists():
                logger.info("Skipping completed point %s SNR=%s", profile, snr)
                continue
            samples_target = int(cfg["eval"].get("samples_per_point", 1000))
            max_samples = int(cfg["eval"].get("max_samples_per_point", samples_target))
            target_errors = int(cfg["eval"].get("target_frame_errors", 100))
            stats = {
                "bp_nms_20": {"errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": []},
                "bp_nms_50": {"errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": []},
                "hybrid_bp_nsg": {"errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": []},
            }
            raw_rows = []
            num = 0
            while num < max_samples:
                num += 1
                frame = simulate_frame(code, float(snr), str(profile), rng)
                true = frame.codeword_internal

                r20 = bp20.decode(frame.llr_internal, collect_trace=False)
                e20 = int(np.any(r20.hard != true))
                stats["bp_nms_20"]["errors"] += e20
                stats["bp_nms_20"]["samples"] += 1
                stats["bp_nms_20"]["queries"].append(0)
                stats["bp_nms_20"]["main_success"] += int(r20.success)
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "bp_nms_20", "block_error": e20, "queries": 0, "action": "bp_success" if r20.success else "bp_fail", "main_success": int(r20.success), "rescue_used": 0, "micro_bp_used": 0})

                r50 = bp50.decode(frame.llr_internal, collect_trace=False)
                e50 = int(np.any(r50.hard != true))
                stats["bp_nms_50"]["errors"] += e50
                stats["bp_nms_50"]["samples"] += 1
                stats["bp_nms_50"]["queries"].append(0)
                stats["bp_nms_50"]["main_success"] += int(r50.success)
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "bp_nms_50", "block_error": e50, "queries": 0, "action": "bp_success" if r50.success else "bp_fail", "main_success": int(r50.success), "rescue_used": 0, "micro_bp_used": 0})

                hr = hybrid.decode(frame.llr_internal, snr_db=float(snr), profile=str(profile), collect_trace=True)
                eh = int(np.any(hr.codeword != true))
                stats["hybrid_bp_nsg"]["errors"] += eh
                stats["hybrid_bp_nsg"]["samples"] += 1
                stats["hybrid_bp_nsg"]["queries"].append(int(hr.queries))
                stats["hybrid_bp_nsg"]["rescue"] += int(hr.rescue_invoked and hr.rescue_success and not hr.main_success)
                stats["hybrid_bp_nsg"]["micro"] += int(hr.used_micro_bp)
                stats["hybrid_bp_nsg"]["main_success"] += int(hr.main_success)
                stats["hybrid_bp_nsg"]["latency"].append(float(hr.elapsed_ms))
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "hybrid_bp_nsg", "block_error": eh, "queries": int(hr.queries), "action": hr.action, "main_success": int(hr.main_success), "rescue_used": int(hr.rescue_invoked), "micro_bp_used": int(hr.used_micro_bp), "latency_ms": float(hr.elapsed_ms)})

                # Stop after min samples if the target decoders have enough errors.
                if num >= samples_target:
                    stop = True
                    for dec in cfg["eval"].get("stop_decoders", []):
                        if stats[dec]["errors"] < target_errors:
                            stop = False
                            break
                    if stop:
                        break

            summary_rows = []
            for dec, st in stats.items():
                samples = max(1, int(st["samples"]))
                row = {
                    "profile": profile,
                    "snr_db": float(snr),
                    "decoder": dec,
                    "samples": samples,
                    "frame_errors": int(st["errors"]),
                    "bler": float(st["errors"]) / samples,
                    "avg_queries": float(np.mean(st["queries"])) if st["queries"] else 0.0,
                    "p95_queries": _percentile(st["queries"], 95),
                    "rescue_rate": float(st["rescue"]) / samples,
                    "micro_bp_rate": float(st["micro"]) / samples,
                    "main_success_rate": float(st["main_success"]) / samples,
                    "avg_latency_ms": float(np.mean(st["latency"])) if st["latency"] else 0.0,
                }
                summary_rows.append(row)
                all_summaries.append(row)

            write_csv(summary_path, summary_rows)
            with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as f:
                fieldnames = ["sample_idx", "profile", "snr_db", "decoder", "block_error", "queries", "action", "main_success", "rescue_used", "micro_bp_used", "latency_ms"]
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for row in raw_rows:
                    w.writerow(row)
            logger.info("Completed profile=%s snr=%s samples=%d summary=%s", profile, snr, num, summary_path)

    if all_summaries:
        write_csv(eval_root / "evaluation_summary.csv", all_summaries)
