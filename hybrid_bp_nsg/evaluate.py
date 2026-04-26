from __future__ import annotations

import csv
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .bp import BPDecodeResult, bp_decode
from .channels import channel_diagnostics, simulate_frame
from .code import LDPCCode, build_code, write_code_summary
from .config import save_resolved_config
from .features import (build_rescue_features, candidate_feature_vector, candidate_pool_from_feature_pack,
                       greedy_syndrome_repair_candidates, syndrome_osd_candidates)
from .model import build_rescue_net


@dataclass
class DecodeResult:
    hard: np.ndarray
    success: bool
    queries: int
    elapsed_ms: float
    action: str
    rescue_used: bool = False
    rescue_success: bool = False
    crc_fail: bool = False
    crc_valid_candidates: int = 0
    parity_valid_candidates: int = 0
    main_success: bool = False
    micro_bp_used: bool = False


def _batch_valid_candidates(code: LDPCCode, base_hard: np.ndarray, masks: List[Tuple[np.ndarray, float, str]], parallel_bs: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, str, bool]], int]:
    out: List[Tuple[np.ndarray, np.ndarray, float, str, bool]] = []
    seen: set[Tuple[int, ...]] = set()
    base_syn = code.syndrome(base_hard).astype(np.uint8)
    h_int = code.h.astype(np.int16, copy=False)
    queries = 0
    clean: List[Tuple[Tuple[int, ...], float, str]] = []
    for idx, score, src in masks:
        key = tuple(sorted(set(int(i) for i in np.asarray(idx).reshape(-1).tolist())))
        if key in seen:
            continue
        seen.add(key)
        clean.append((key, float(score), str(src)))
    for start in range(0, len(clean), max(1, int(parallel_bs))):
        chunk = clean[start:start+max(1, int(parallel_bs))]
        if not chunk:
            continue
        dense = np.zeros((len(chunk), code.n), dtype=np.uint8)
        for bi, (key, _score, _src) in enumerate(chunk):
            if key:
                dense[bi, list(key)] = 1
        delta = (h_int @ dense.T.astype(np.int16)) & 1
        valid = np.all(delta == base_syn[:, None], axis=0)
        cand_batch = (base_hard[None, :] ^ dense).astype(np.uint8)
        queries += len(chunk)
        for bi, ok in enumerate(valid.tolist()):
            if ok:
                key, score, src = chunk[bi]
                cand = cand_batch[bi]
                out.append((dense[bi].copy(), cand.copy(), score, src, code.crc_check_internal(cand)))
    return out, queries


def _hybrid_decode(code: LDPCCode, cfg: Dict[str, Any], llr: np.ndarray, profile: str, snr_db: float, model=None, tf=None) -> DecodeResult:
    start = time.perf_counter()
    bcfg = cfg.get("bp", {})
    rcfg = cfg.get("rescue", {})
    mcfg = cfg.get("model", {})
    main = bp_decode(code, llr, iterations=int(bcfg.get("hybrid_main_iterations", 20)), nms_alpha=float(bcfg.get("nms_alpha", 0.8)), early_stop=bool(bcfg.get("early_stop", True)), collect_trace=True)
    if main.success:
        return DecodeResult(hard=main.hard, success=True, queries=0, elapsed_ms=(time.perf_counter()-start)*1e3, action="bp_success", main_success=True, crc_valid_candidates=1, parity_valid_candidates=1)
    fp = build_rescue_features(code, llr, main, snr_db, profile, num_segments=int(mcfg.get("num_segments", 8)), target_basis=str(rcfg.get("target_basis", "bp")))
    bit_cost = np.asarray(fp["bit_cost"], dtype=np.float32).copy()
    weight_order = list(range(1, min(int(rcfg.get("max_standard_weight", 10)), 4) + 1))
    packet_emb = None
    candidate_net = None
    if model is not None and tf is not None:
        inp = {
            "var_features": tf.convert_to_tensor(fp["var_features"][None, ...], dtype=tf.float32),
            "check_features": tf.convert_to_tensor(fp["check_features"][None, ...], dtype=tf.float32),
            "global_features": tf.convert_to_tensor(fp["global_features"][None, ...], dtype=tf.float32),
            "candidate_features": tf.zeros([1, int(mcfg.get("rerank_list_size", 12)), 14], dtype=tf.float32),
        }
        out = model(inp, training=False)
        bit_p = tf.sigmoid(out["bit_logits"])[0].numpy().astype(np.float32)
        bit_cost = bit_cost - 1.50 * (bit_p - np.mean(bit_p)).astype(np.float32)
        fp = dict(fp)
        fp["bit_cost"] = bit_cost
        fp["heuristic_order"] = np.argsort(bit_cost).astype(np.int16)
        w_logits = out["weight_logits"][0].numpy()
        weight_order = [int(x) for x in np.argsort(-w_logits)[: int(rcfg.get("likely_weight_topk", 10))] if 0 < int(x) <= int(rcfg.get("max_expanded_weight", 32))]
        if not weight_order:
            weight_order = list(range(1, min(int(rcfg.get("max_standard_weight", 10)), 4) + 1))
        packet_emb = out["packet_embedding"]
        candidate_net = model.reranker if bool(rcfg.get("rerank_use_net", True)) else None
    base_hard = np.asarray(fp["grand_base_hard"], dtype=np.uint8)
    masks: List[Tuple[np.ndarray, float, str]] = []
    masks.extend(candidate_pool_from_feature_pack(
        fp,
        max_weight=min(4, int(rcfg.get("max_expanded_weight", 32))),
        budget=int(rcfg.get("direct_budget", 2200)),
        pool_size=int(rcfg.get("top_k_bits", 128)),
        combo_pool_w_le3=int(rcfg.get("combo_pool_w_le3", 28)),
        combo_pool_w_gt3=int(rcfg.get("combo_pool_w_gt3", 18)),
    ))
    if bool(rcfg.get("enable_greedy_repair", True)):
        masks.extend(greedy_syndrome_repair_candidates(code, base_hard, bit_cost, max_steps=int(rcfg.get("greedy_repair_steps", 48)), max_candidates=int(rcfg.get("greedy_repair_candidates", 8))))
    if bool(rcfg.get("enable_osd_repair", True)):
        masks.extend(syndrome_osd_candidates(code, base_hard, bit_cost, support_sizes=rcfg.get("osd_support_sizes", [64,96,128,160]), jitter_passes=int(rcfg.get("osd_jitter_passes", 2))))
    masks = sorted(masks, key=lambda x: float(x[1]))[: int(rcfg.get("expanded_budget", 3600))]
    valid, queries = _batch_valid_candidates(code, base_hard, masks, int(rcfg.get("parallel_test_batch_size", 256)))
    parity_valid = len(valid)
    crc_valid = [v for v in valid if v[4] or not code.has_outer_crc]
    require_crc = bool(rcfg.get("require_crc_for_accept", True)) and code.has_outer_crc
    candidate_set = crc_valid if (require_crc and crc_valid) else valid
    if not candidate_set:
        return DecodeResult(hard=main.hard, success=False, queries=queries, elapsed_ms=(time.perf_counter()-start)*1e3, action="no_crc_valid_candidate" if require_crc else "rescue_fail", rescue_used=True, crc_fail=True, crc_valid_candidates=len(crc_valid), parity_valid_candidates=parity_valid)
    scores = np.array([-v[2] for v in candidate_set], dtype=np.float32)
    if candidate_net is not None and packet_emb is not None and tf is not None:
        cand_feats = []
        for mask, cand, prior, _src, crc_ok in candidate_set:
            cand_feats.append(candidate_feature_vector(code, llr, base_hard, cand, mask, fp, prior, crc_ok))
        cand_feats_tf = tf.convert_to_tensor(np.stack(cand_feats, axis=0)[None, ...], dtype=tf.float32)
        net_scores = candidate_net(packet_emb, cand_feats_tf, training=False).numpy()[0]
        scores = net_scores + float(rcfg.get("rerank_prior_scale", 0.1)) * scores
    best_i = int(np.argmax(scores))
    best = candidate_set[best_i]
    hard = best[1]
    ok = bool(code.is_codeword(hard) and (code.crc_check_internal(hard) or not require_crc))
    return DecodeResult(hard=hard, success=ok, queries=queries, elapsed_ms=(time.perf_counter()-start)*1e3, action=f"rescue_success_{best[3]}" if ok else "reject_crc_fail", rescue_used=True, rescue_success=ok, crc_fail=not ok, crc_valid_candidates=len(crc_valid), parity_valid_candidates=parity_valid)


def _bp_decoder_result(code: LDPCCode, llr: np.ndarray, iterations: int, cfg: Dict[str, Any]) -> DecodeResult:
    start = time.perf_counter()
    bp = bp_decode(code, llr, iterations=iterations, nms_alpha=float(cfg.get("bp", {}).get("nms_alpha", 0.8)), early_stop=bool(cfg.get("bp", {}).get("early_stop", True)), collect_trace=False)
    return DecodeResult(hard=bp.hard, success=bp.success, queries=0, elapsed_ms=(time.perf_counter()-start)*1e3, action="bp_success" if bp.success else "bp_fail", main_success=bp.success, crc_fail=not bp.success)


def evaluate(cfg: Dict[str, Any]) -> None:
    import tensorflow as tf
    ecfg = cfg.get("eval", {})
    if bool(ecfg.get("require_gpu", False)) and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("eval.require_gpu=true but TensorFlow sees no GPU")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(ecfg.get("cpu_threads", 0)) or 0)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(8, int(ecfg.get("cpu_threads", 8)))))
    except Exception:
        pass
    out_dir = Path(cfg["project"]["output_dir"])
    code = build_code(cfg)
    write_code_summary(code, out_dir)
    save_resolved_config(cfg, out_dir)
    print("code_family:", code.family)
    print("transport_k:", code.transport_k)
    print("ldpc_k:", code.k)
    print("n_internal:", code.n)
    print("n_transmitted:", code.n_transmitted)
    print("profiles:", ecfg.get("profiles", ["AWGN"]))
    for line in channel_diagnostics(code, cfg, list(ecfg.get("profiles", ["AWGN"]))):
        print(line)
    model = build_rescue_net(code, cfg)
    dummy = {
        "var_features": tf.zeros([1, code.n, 16], dtype=tf.float32),
        "check_features": tf.zeros([1, code.m, 6], dtype=tf.float32),
        "global_features": tf.zeros([1, 9], dtype=tf.float32),
        "candidate_features": tf.zeros([1, int(cfg.get("model", {}).get("rerank_list_size", 12)), 14], dtype=tf.float32),
    }
    _ = model(dummy, training=False)
    weights = out_dir / "checkpoints" / "rescue_net_tf.best.weights.h5"
    if not weights.exists():
        weights = out_dir / "checkpoints" / "rescue_net_tf.weights.h5"
    if weights.exists():
        model.load_weights(str(weights))
        print(f"Loaded TensorFlow rescue network using weights={weights}")
    else:
        print("WARNING: no rescue weights found; evaluating untrained neural policy")
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.get("project", {}).get("seed", 31415)) + 999)
    decoders = list(ecfg.get("stop_decoders", ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"]))
    for profile in ecfg.get("profiles", ["AWGN"]):
        for snr in ecfg.get("snr_db_grid", [0, 2, 4]):
            samples_target = int(ecfg.get("samples_per_point", 4000))
            stats = {d: {"samples": 0, "frame_errors": 0, "payload_frame_errors": 0, "ber_internal": 0.0, "payload_ber": 0.0,
                         "queries": [], "lat": [], "rescue": 0, "micro": 0, "main_success": 0, "crc_fail": 0, "crc_valid": [], "parity_valid": []} for d in decoders}
            for _ in range(samples_target):
                frame = simulate_frame(code, float(snr), str(profile), rng, cfg)
                true = frame.codeword_internal
                results: Dict[str, DecodeResult] = {}
                if "bp_nms_20" in decoders:
                    results["bp_nms_20"] = _bp_decoder_result(code, frame.llr_internal, int(cfg.get("bp", {}).get("hybrid_main_iterations", 20)), cfg)
                if "bp_nms_50" in decoders:
                    results["bp_nms_50"] = _bp_decoder_result(code, frame.llr_internal, int(cfg.get("bp", {}).get("strong_iterations", 50)), cfg)
                if "hybrid_bp_nsg" in decoders:
                    results["hybrid_bp_nsg"] = _hybrid_decode(code, cfg, frame.llr_internal, str(profile), float(snr), model=model, tf=tf)
                for d, res in results.items():
                    st = stats[d]
                    st["samples"] += 1
                    err_bits = (res.hard.astype(np.uint8) ^ true.astype(np.uint8))
                    frame_err = int(err_bits.any())
                    st["frame_errors"] += frame_err
                    st["payload_frame_errors"] += int(err_bits[: code.transport_k].any())
                    st["ber_internal"] += float(err_bits.mean())
                    st["payload_ber"] += float(err_bits[: code.transport_k].mean())
                    st["queries"].append(float(res.queries))
                    st["lat"].append(float(res.elapsed_ms))
                    st["rescue"] += int(res.rescue_success)
                    st["micro"] += int(res.micro_bp_used)
                    st["main_success"] += int(res.main_success)
                    st["crc_fail"] += int(res.crc_fail)
                    st["crc_valid"].append(float(res.crc_valid_candidates))
                    st["parity_valid"].append(float(res.parity_valid_candidates))
            prof_dir = out_dir / "evaluation" / f"profile_{profile}" / f"snr_{float(snr):+0.1f}dB"
            prof_dir.mkdir(parents=True, exist_ok=True)
            point_rows = []
            for d in decoders:
                st = stats[d]
                s = max(1, int(st["samples"]))
                row = {
                    "profile": profile,
                    "snr_db": float(snr),
                    "decoder": d,
                    "samples": s,
                    "frame_errors": int(st["frame_errors"]),
                    "payload_frame_errors": int(st["payload_frame_errors"]),
                    "fer": float(st["frame_errors"]) / s,
                    "bler": float(st["frame_errors"]) / s,
                    "payload_fer": float(st["payload_frame_errors"]) / s,
                    "ber_internal": float(st["ber_internal"]) / s,
                    "payload_ber": float(st["payload_ber"]) / s,
                    "avg_queries": float(np.mean(st["queries"])) if st["queries"] else 0.0,
                    "p95_queries": float(np.percentile(st["queries"], 95)) if st["queries"] else 0.0,
                    "rescue_rate": float(st["rescue"]) / s,
                    "micro_bp_rate": float(st["micro"]) / s,
                    "main_success_rate": float(st["main_success"]) / s,
                    "avg_latency_ms": float(np.mean(st["lat"])) if st["lat"] else 0.0,
                    "crc_fail_rate": float(st["crc_fail"]) / s,
                    "framework": "tensorflow",
                    "num_gpus": len(tf.config.list_physical_devices("GPU")),
                    "weights_used": str(weights) if weights.exists() else "",
                    "channel_type": "AWGN" if str(profile).upper()=="AWGN" else "SIONNA_CDL_C_SURROGATE",
                    "modulation": "QPSK",
                    "perfect_csi": True,
                    "equalizer": "identity" if str(profile).upper()=="AWGN" else "one_tap_perfect_csi",
                    "avg_crc_valid_candidates": float(np.mean(st["crc_valid"])) if st["crc_valid"] else 0.0,
                    "avg_parity_valid_candidates": float(np.mean(st["parity_valid"])) if st["parity_valid"] else 0.0,
                }
                rows.append(row); point_rows.append(row)
            with (prof_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
                writer.writeheader(); writer.writerows(point_rows)
            print(f"Completed profile={profile} snr={snr} samples={samples_target} summary={prof_dir/'summary.csv'}", flush=True)
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
