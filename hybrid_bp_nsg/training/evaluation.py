from __future__ import annotations

import csv
import os
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..channels.simulator import simulate_frame
from ..codes.factory import build_code
from ..decoders.bp import BeliefPropagationDecoder
from ..decoders.hybrid import HybridBPNSGDecoder
from ..models.rescue_net import build_rescue_net_from_shapes
from ..training.dataset import infer_shapes
from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logging import get_logger
from ..utils.tf_gpu import configure_tensorflow, tensorflow_device_string


def _dummy_build_model(tf, model, shapes: Dict[str, int]) -> None:
    b = 1
    model({
        "var_features": tf.zeros((b, int(shapes["n"]), int(shapes["num_var_features"])), dtype=tf.float32),
        "check_features": tf.zeros((b, int(shapes["m"]), int(shapes["num_check_features"])), dtype=tf.float32),
        "global_features": tf.zeros((b, int(shapes["num_global_features"])), dtype=tf.float32),
        "heuristic_order": tf.zeros((b, int(shapes["n"])), dtype=tf.int32),
        "candidate_features": tf.zeros((b, max(1, int(shapes.get("num_candidates", shapes.get("rerank_list_size", 12)))), int(shapes.get("candidate_feature_dim", 14))), dtype=tf.float32),
    }, training=False)


def _load_rescue_net(cfg: Dict[str, object], code):
    tf, gpus = configure_tensorflow(
        require_gpu=bool(cfg.get("eval", {}).get("require_gpu", False)),
        mixed_precision=bool(cfg.get("eval", {}).get("mixed_precision", False)),
        xla=bool(cfg.get("eval", {}).get("xla", True)),
        cpu_threads=int(cfg.get("eval", {}).get("cpu_threads", os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)),
    )
    ckpt_dir = Path(cfg["project"]["output_dir"]) / "checkpoints"
    best_weights = ckpt_dir / "rescue_net_tf.best.weights.h5"
    final_weights = ckpt_dir / "rescue_net_tf.weights.h5"
    meta_path = ckpt_dir / "rescue_net_tf_meta.json"
    weights_path = best_weights if best_weights.exists() else final_weights
    if not weights_path.exists():
        return None, tf, gpus, None

    shapes = None
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            shapes = meta.get("shapes")
        except Exception:
            shapes = None
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
                "candidate_feature_dim": 14,
            }
    model = build_rescue_net_from_shapes(shapes, cfg, code)
    _dummy_build_model(tf, model, shapes)
    model.load_weights(str(weights_path))
    return model, tf, gpus, {"weights_path": str(weights_path), "meta": meta or {}}


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def evaluate(cfg: Dict[str, object]) -> None:
    logger = get_logger("evaluate")
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    eval_root = ensure_dir(out_dir / "evaluation")
    write_json(cfg, out_dir / "artifacts" / "resolved_config.json")
    tf, gpus = configure_tensorflow(
        require_gpu=bool(cfg.get("eval", {}).get("require_gpu", False)),
        mixed_precision=bool(cfg.get("eval", {}).get("mixed_precision", False)),
        xla=bool(cfg.get("eval", {}).get("xla", True)),
        cpu_threads=int(cfg.get("eval", {}).get("cpu_threads", os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)),
    )
    code = build_code(cfg["code"])

    rescue_net, _, gpus, net_meta = _load_rescue_net(cfg, code)
    device = tensorflow_device_string()
    if rescue_net is None:
        logger.info("No trained rescue network found; hybrid uses heuristic policy.")
    else:
        logger.info("Loaded TensorFlow rescue network on %s with GPUs=%s using weights=%s", device, gpus, net_meta["weights_path"])

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
    profiles = list(cfg["eval"].get("profiles", ["AWGN"]))
    snrs = list(cfg["eval"].get("snr_db_grid", [0]))
    for profile in profiles:
        for snr in snrs:
            point_dir = ensure_dir(eval_root / f"profile_{profile}" / f"snr_{float(snr):+0.1f}dB")
            summary_path = point_dir / "summary.csv"
            raw_path = point_dir / "raw_records.csv.gz"
            samples_target = int(cfg["eval"].get("samples_per_point", 1000))
            max_samples = int(cfg["eval"].get("max_samples_per_point", samples_target))
            target_errors = int(cfg["eval"].get("target_frame_errors", 100))
            point_channel_meta = None
            stats = {
                "bp_nms_20": {"errors": 0, "payload_errors": 0, "bit_errors": 0, "payload_bit_errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": [], "crc_fail": 0},
                "bp_nms_50": {"errors": 0, "payload_errors": 0, "bit_errors": 0, "payload_bit_errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": [], "crc_fail": 0},
                "hybrid_bp_nsg": {"errors": 0, "payload_errors": 0, "bit_errors": 0, "payload_bit_errors": 0, "samples": 0, "queries": [], "rescue": 0, "micro": 0, "main_success": 0, "latency": [], "crc_fail": 0, "crc_valid_cands": [], "parity_valid_cands": []},
            }
            raw_rows = []
            num = 0
            while num < max_samples:
                num += 1
                frame = simulate_frame(code, float(snr), str(profile), rng, channel_cfg=cfg.get("channel", {}))
                if point_channel_meta is None:
                    point_channel_meta = dict(frame.channel_meta)
                true = np.asarray(frame.codeword_internal, dtype=np.uint8).reshape(-1)
                payload_true = np.asarray(frame.message, dtype=np.uint8).reshape(-1)

                t0 = time.perf_counter()
                r20 = bp20.decode(frame.llr_internal, collect_trace=False)
                t20 = (time.perf_counter() - t0) * 1e3
                hard20 = np.asarray(r20.hard, dtype=np.uint8).reshape(-1)
                pay20 = np.asarray(code.payload_bits(hard20), dtype=np.uint8).reshape(-1)
                e20 = int(np.any(hard20 != true))
                pe20 = int(np.any(pay20 != payload_true))
                be20 = int(np.sum(hard20 != true))
                pbe20 = int(np.sum(pay20 != payload_true))
                stats["bp_nms_20"]["errors"] += e20
                stats["bp_nms_20"]["payload_errors"] += pe20
                stats["bp_nms_20"]["bit_errors"] += be20
                stats["bp_nms_20"]["payload_bit_errors"] += pbe20
                stats["bp_nms_20"]["samples"] += 1
                stats["bp_nms_20"]["queries"].append(0)
                stats["bp_nms_20"]["main_success"] += int(r20.success)
                stats["bp_nms_20"]["latency"].append(float(t20))
                stats["bp_nms_20"]["crc_fail"] += int(not r20.crc_ok)
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "bp_nms_20", "frame_error": e20, "payload_frame_error": pe20, "bit_errors": be20, "payload_bit_errors": pbe20, "queries": 0, "action": "bp_success" if r20.success else ("bp_crc_fail" if r20.crc_ok is False and int(r20.syndrome.sum()) == 0 else "bp_fail"), "main_success": int(r20.success), "rescue_used": 0, "micro_bp_used": 0, "latency_ms": float(t20), "crc_ok": int(r20.crc_ok)})

                t0 = time.perf_counter()
                r50 = bp50.decode(frame.llr_internal, collect_trace=False)
                t50 = (time.perf_counter() - t0) * 1e3
                hard50 = np.asarray(r50.hard, dtype=np.uint8).reshape(-1)
                pay50 = np.asarray(code.payload_bits(hard50), dtype=np.uint8).reshape(-1)
                e50 = int(np.any(hard50 != true))
                pe50 = int(np.any(pay50 != payload_true))
                be50 = int(np.sum(hard50 != true))
                pbe50 = int(np.sum(pay50 != payload_true))
                stats["bp_nms_50"]["errors"] += e50
                stats["bp_nms_50"]["payload_errors"] += pe50
                stats["bp_nms_50"]["bit_errors"] += be50
                stats["bp_nms_50"]["payload_bit_errors"] += pbe50
                stats["bp_nms_50"]["samples"] += 1
                stats["bp_nms_50"]["queries"].append(0)
                stats["bp_nms_50"]["main_success"] += int(r50.success)
                stats["bp_nms_50"]["latency"].append(float(t50))
                stats["bp_nms_50"]["crc_fail"] += int(not r50.crc_ok)
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "bp_nms_50", "frame_error": e50, "payload_frame_error": pe50, "bit_errors": be50, "payload_bit_errors": pbe50, "queries": 0, "action": "bp_success" if r50.success else ("bp_crc_fail" if r50.crc_ok is False and int(r50.syndrome.sum()) == 0 else "bp_fail"), "main_success": int(r50.success), "rescue_used": 0, "micro_bp_used": 0, "latency_ms": float(t50), "crc_ok": int(r50.crc_ok)})

                hr = hybrid.decode(frame.llr_internal, snr_db=float(snr), profile=str(profile), collect_trace=True)
                hardh = np.asarray(hr.codeword, dtype=np.uint8).reshape(-1)
                payh = np.asarray(code.payload_bits(hardh), dtype=np.uint8).reshape(-1)
                eh = int(np.any(hardh != true))
                peh = int(np.any(payh != payload_true))
                beh = int(np.sum(hardh != true))
                pbeh = int(np.sum(payh != payload_true))
                stats["hybrid_bp_nsg"]["errors"] += eh
                stats["hybrid_bp_nsg"]["payload_errors"] += peh
                stats["hybrid_bp_nsg"]["bit_errors"] += beh
                stats["hybrid_bp_nsg"]["payload_bit_errors"] += pbeh
                stats["hybrid_bp_nsg"]["samples"] += 1
                stats["hybrid_bp_nsg"]["queries"].append(int(hr.queries))
                stats["hybrid_bp_nsg"]["rescue"] += int(hr.rescue_invoked and hr.rescue_success and not hr.main_success)
                stats["hybrid_bp_nsg"]["micro"] += int(hr.used_micro_bp)
                stats["hybrid_bp_nsg"]["main_success"] += int(hr.main_success)
                stats["hybrid_bp_nsg"]["latency"].append(float(hr.elapsed_ms))
                stats["hybrid_bp_nsg"]["crc_fail"] += int(not hr.crc_ok)
                stats["hybrid_bp_nsg"]["crc_valid_cands"].append(int(hr.crc_valid_candidates))
                stats["hybrid_bp_nsg"]["parity_valid_cands"].append(int(hr.parity_valid_candidates))
                raw_rows.append({"sample_idx": num, "profile": profile, "snr_db": snr, "decoder": "hybrid_bp_nsg", "frame_error": eh, "payload_frame_error": peh, "bit_errors": beh, "payload_bit_errors": pbeh, "queries": int(hr.queries), "action": hr.action, "main_success": int(hr.main_success), "rescue_used": int(hr.rescue_invoked), "micro_bp_used": int(hr.used_micro_bp), "latency_ms": float(hr.elapsed_ms), "crc_ok": int(hr.crc_ok), "crc_valid_candidates": int(hr.crc_valid_candidates), "parity_valid_candidates": int(hr.parity_valid_candidates)})

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
                    "payload_frame_errors": int(st["payload_errors"]),
                    "fer": float(st["errors"]) / samples,
                    "bler": float(st["errors"]) / samples,
                    "payload_fer": float(st["payload_errors"]) / samples,
                    "ber_internal": float(st["bit_errors"]) / max(1, samples * code.n),
                    "payload_ber": float(st["payload_bit_errors"]) / max(1, samples * int(getattr(code, "transport_k", code.k))),
                    "avg_queries": float(np.mean(st["queries"])) if st["queries"] else 0.0,
                    "p95_queries": _percentile(st["queries"], 95),
                    "rescue_rate": float(st["rescue"]) / samples,
                    "micro_bp_rate": float(st["micro"]) / samples,
                    "main_success_rate": float(st["main_success"]) / samples,
                    "avg_latency_ms": float(np.mean(st["latency"])) if st["latency"] else 0.0,
                    "crc_fail_rate": float(st["crc_fail"]) / samples,
                    "framework": "tensorflow",
                    "num_gpus": len(gpus),
                    "weights_used": net_meta["weights_path"] if net_meta else "heuristic_only",
                    "channel_type": (point_channel_meta or {}).get("channel_type", str(profile)),
                    "modulation": (point_channel_meta or {}).get("modulation", "QPSK"),
                    "perfect_csi": (point_channel_meta or {}).get("perfect_csi", True),
                    "equalizer": (point_channel_meta or {}).get("equalizer", "unknown"),
                }
                if dec == "hybrid_bp_nsg":
                    row["avg_crc_valid_candidates"] = float(np.mean(st["crc_valid_cands"])) if st["crc_valid_cands"] else 0.0
                    row["avg_parity_valid_candidates"] = float(np.mean(st["parity_valid_cands"])) if st["parity_valid_cands"] else 0.0
                summary_rows.append(row)
                all_summaries.append(row)

            write_csv(summary_path, summary_rows)
            with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "sample_idx", "profile", "snr_db", "decoder", "frame_error", "payload_frame_error", "bit_errors", "payload_bit_errors", "queries", "action",
                    "main_success", "rescue_used", "micro_bp_used", "latency_ms", "crc_ok", "crc_valid_candidates", "parity_valid_candidates"
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for row in raw_rows:
                    w.writerow(row)
            logger.info("Completed profile=%s snr=%s samples=%d summary=%s", profile, snr, num, summary_path)


    if all_summaries:
        write_csv(eval_root / "evaluation_summary.csv", all_summaries)
