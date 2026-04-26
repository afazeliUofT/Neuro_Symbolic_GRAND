from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _force_cpu_for_generation() -> None:
    """Hide CUDA before TensorFlow/Sionna imports in generation workers."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _force_single_thread_worker() -> None:
    """Prevent 8/32 generation workers from each using all BLAS/TF threads."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = "1"


def _choose(rng: np.random.Generator, vals: List[Any], probs: List[float] | None):
    if probs is None:
        return vals[int(rng.integers(0, len(vals)))]
    p = np.asarray(probs, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    return vals[int(rng.choice(len(vals), p=p))]


def _generate_one(code, cfg: Dict[str, Any], rng: np.random.Generator, split: str) -> Dict[str, Any] | None:
    # Import these only after the worker has hidden CUDA. This prevents TensorFlow/Sionna
    # from touching the GPU during dataset generation on GPU smoke jobs.
    from .bp import bp_decode
    from .channels import simulate_frame
    from .features import build_rescue_features, build_training_labels

    dcfg = cfg.get("data", {})
    rcfg = cfg.get("rescue", {})
    mcfg = cfg.get("model", {})
    bcfg = cfg.get("bp", {})
    profiles = list(dcfg.get("profiles", ["AWGN"]))
    profile_probs = dcfg.get("profile_probs", None)
    snrs = list(dcfg.get("failed_snr_db_grid", [2, 3, 4]))
    snr_probs = dcfg.get("failed_snr_probs", None)
    profile = str(_choose(rng, profiles, profile_probs))
    snr_db = float(_choose(rng, snrs, snr_probs))

    frame = simulate_frame(code, snr_db, profile, rng, cfg)
    bp = bp_decode(
        code,
        frame.llr_internal,
        iterations=int(dcfg.get("bp_collect_iterations", bcfg.get("hybrid_main_iterations", 20))),
        nms_alpha=float(bcfg.get("nms_alpha", 0.8)),
        early_stop=bool(bcfg.get("early_stop", True)),
        collect_trace=True,
    )
    if bool(dcfg.get("sample_failed_only", True)) and bp.success:
        return None

    fp = build_rescue_features(
        code,
        frame.llr_internal,
        bp,
        snr_db,
        profile,
        num_segments=int(mcfg.get("num_segments", 8)),
        target_basis=str(rcfg.get("target_basis", "bp")),
    )
    labels = build_training_labels(code, fp, frame.codeword_internal, bp, rcfg, mcfg, llr_internal=frame.llr_internal)
    tw = int(np.asarray(labels["target_weight"]).reshape(-1)[0])

    if bool(dcfg.get("filter_by_target_weight", False)):
        if tw < int(dcfg.get("min_target_weight", 1)) or tw > int(dcfg.get("max_target_weight", rcfg.get("max_expanded_weight", 32))):
            return None
    if bool(dcfg.get("require_candidate_positive", False)):
        pos = int(np.sum(labels["candidate_labels"].astype(np.uint8) * labels["candidate_valid"].astype(np.uint8)))
        if pos <= 0:
            return None
    if bool(dcfg.get("require_reachable", False)):
        if int(labels["standard_reachable"]) == 0 and int(labels["expanded_reachable"]) == 0:
            return None

    return {
        "var_features": fp["var_features"].astype(np.float16),
        "check_features": fp["check_features"].astype(np.float16),
        "global_features": fp["global_features"].astype(np.float16),
        "heuristic_order": fp["heuristic_order"].astype(np.int16),
        "bit_labels": labels["bit_labels"].astype(np.uint8),
        "segment_labels": labels["segment_labels"].astype(np.uint8),
        "weight_label": labels["weight_label"].astype(np.int16),
        "standard_reachable": labels["standard_reachable"].astype(np.uint8),
        "expanded_reachable": labels["expanded_reachable"].astype(np.uint8),
        "rescueable": labels["rescueable"].astype(np.uint8),
        "candidate_features": labels["candidate_features"].astype(np.float16),
        "candidate_labels": labels["candidate_labels"].astype(np.uint8),
        "candidate_valid": labels["candidate_valid"].astype(np.uint8),
        "profile_id": fp["profile_id"].astype(np.int16),
        "snr_db": fp["snr_db"].astype(np.float16),
        "bp_success": np.array(int(bp.success), dtype=np.uint8),
        "target_weight": labels["target_weight"].astype(np.int16),
        "bp_residual_weight": labels["bp_residual_weight"].astype(np.int16),
        "transport_k": np.array(int(code.transport_k), dtype=np.int16),
    }


def _stack_rows(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    keys = list(rows[0].keys())
    return {k: np.stack([r[k] for r in rows], axis=0) for k in keys}


def _worker_chunk(args: Tuple[Dict[str, Any], str, int, int, int, str]) -> Dict[str, Any]:
    _force_cpu_for_generation()
    _force_single_thread_worker()
    from .code import build_code

    cfg, split, want, worker_id, seed, out_dir = args
    rng = np.random.default_rng(seed)
    code = build_code(cfg)
    out_path = Path(out_dir) / "datasets" / split
    out_path.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    attempted = 0
    max_attempts = int(max(want, want * int(cfg.get("data", {}).get("max_attempt_multiplier", 80))))
    while len(rows) < want and attempted < max_attempts:
        attempted += 1
        item = _generate_one(code, cfg, rng, split)
        if item is not None:
            rows.append(item)
    if rows:
        arrs = _stack_rows(rows)
        fname = out_path / f"{split}_w{worker_id:03d}_0000.npz"
        np.savez_compressed(fname, **arrs)
    else:
        fname = out_path / f"{split}_w{worker_id:03d}_EMPTY.txt"
        fname.write_text("no rows kept\n", encoding="utf-8")
    return {"split": split, "kept": len(rows), "attempted": attempted, "shards": int(bool(rows)), "prefix": f"w{worker_id:03d}"}


def _chunk_plan(total: int, shard: int) -> List[int]:
    out = []
    rem = int(total)
    while rem > 0:
        k = min(int(shard), rem)
        out.append(k)
        rem -= k
    return out


def generate(cfg: Dict[str, Any]) -> None:
    _force_cpu_for_generation()

    from .channels import channel_diagnostics
    from .code import build_code, write_code_summary
    from .config import save_resolved_config

    out_dir = Path(cfg["project"]["output_dir"])
    (out_dir / "datasets" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets" / "val").mkdir(parents=True, exist_ok=True)
    code = build_code(cfg)
    write_code_summary(code, out_dir)
    save_resolved_config(cfg, out_dir)
    print("code_family:", code.family)
    print("transport_k:", code.transport_k)
    print("ldpc_k:", code.k)
    print("n_internal:", code.n)
    print("n_transmitted:", code.n_transmitted)
    print("profiles:", cfg.get("data", {}).get("profiles", ["AWGN"]))
    for line in channel_diagnostics(code, cfg, list(cfg.get("data", {}).get("profiles", ["AWGN"]))):
        print(line)

    dcfg = cfg.get("data", {})
    shard = int(dcfg.get("shard_size", 250))
    tasks: List[Tuple[Dict[str, Any], str, int, int, int, str]] = []
    wid = 0
    seed0 = int(cfg.get("project", {}).get("seed", 31415))
    for split, total in [("train", int(dcfg.get("train_samples", 0))), ("val", int(dcfg.get("val_samples", 0)))]:
        for want in _chunk_plan(total, shard):
            tasks.append((cfg, split, want, wid, seed0 + 1009 * wid + (0 if split == "train" else 777777), str(out_dir)))
            wid += 1

    num_workers = max(1, int(dcfg.get("num_workers", 1)))
    print(f"Parallel generation with {num_workers} workers")
    kept = {"train": 0, "val": 0}
    attempted = 0

    if num_workers == 1:
        for t in tasks:
            r = _worker_chunk(t)
            print("Completed worker chunk:", r, flush=True)
            kept[r["split"]] += int(r["kept"])
            attempted += int(r["attempted"])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker_chunk, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                print("Completed worker chunk:", r, flush=True)
                kept[r["split"]] += int(r["kept"])
                attempted += int(r["attempted"])

    print(f"Parallel generation complete: attempted={attempted} kept={kept}")
    (out_dir / "artifacts" / "generation_summary.json").write_text(
        json.dumps({"attempted": attempted, "kept": kept}, indent=2), encoding="utf-8"
    )
