from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..channels.simulator import simulate_frame
from ..codes.factory import build_code
from ..decoders.bp import BeliefPropagationDecoder
from ..training.features import build_rescue_features, build_training_labels
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from ..config import normalize_snr_sampling


def _choose_snr_profile(cfg: Dict[str, object], rng: np.random.Generator):
    normalize_snr_sampling(cfg.setdefault("data", {}))
    grid = np.asarray(cfg["data"].get("failed_snr_db_grid", [0, 1, 2]), dtype=np.float32)
    probs = np.asarray(cfg["data"].get("failed_snr_probs", np.ones(len(grid))), dtype=np.float64)
    if probs.size != grid.size:
        raise RuntimeError(f"internal config normalization failed: len(failed_snr_db_grid)={grid.size} len(failed_snr_probs)={probs.size}")
    probs = probs / max(float(probs.sum()), 1e-12)
    snr = float(rng.choice(grid, p=probs))
    profiles = list(cfg["data"].get("profiles", ["A"]))
    profile = str(rng.choice(profiles))
    return snr, profile


def _flush_shard(path: Path, rows: List[Dict[str, np.ndarray]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys = rows[0].keys()
    arrays = {}
    for k in keys:
        vals = [r[k] for r in rows]
        arr = np.stack(vals, axis=0)
        if arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        arrays[k] = arr
    np.savez_compressed(path, **arrays)


def _make_row(cfg: Dict[str, object], code, bp, rng: np.random.Generator):
    while True:
        snr, profile = _choose_snr_profile(cfg, rng)
        frame = simulate_frame(code, snr, profile, rng, channel_cfg=cfg.get("channel", {}))
        bp_result = bp.decode(frame.llr_internal, collect_trace=True)
        if bool(cfg["data"].get("sample_failed_only", True)) and bp_result.success:
            continue
        fp = build_rescue_features(
            code, frame.llr_internal, bp_result, snr, profile,
            num_segments=int(cfg["model"]["num_segments"]),
            target_basis=str(cfg["rescue"].get("target_basis", "channel_with_bp_punctures")),
        )
        labels = build_training_labels(code, fp, frame.codeword_internal, bp_result, cfg["rescue"], cfg["model"], llr_internal=frame.llr_internal)
        row = {
            "var_features": fp["var_features"].astype(np.float32),
            "check_features": fp["check_features"].astype(np.float32),
            "global_features": fp["global_features"].astype(np.float32),
            "heuristic_order": fp["heuristic_order"].astype(np.int16),
            "bit_labels": labels["bit_labels"].astype(np.uint8),
            "segment_labels": labels["segment_labels"].astype(np.uint8),
            "weight_label": labels["weight_label"].astype(np.int16),
            "standard_reachable": labels["standard_reachable"].astype(np.uint8),
            "expanded_reachable": labels["expanded_reachable"].astype(np.uint8),
            "rescueable": labels["rescueable"].astype(np.uint8),
            "candidate_features": labels["candidate_features"].astype(np.float32),
            "candidate_labels": labels["candidate_labels"].astype(np.uint8),
            "candidate_valid": labels["candidate_valid"].astype(np.uint8),
            "profile_id": np.array({"A": 0, "AWGN": 0, "AWGN_QPSK": 0, "CDL_C": 1, "CDLC": 1, "CDL-C": 1, "C": 1, "B": 2, "D": 3, "E": 4}.get(str(profile).upper(), 0), dtype=np.int16),
            "snr_db": np.array(snr, dtype=np.float32),
            "bp_success": np.array(int(bp_result.success), dtype=np.uint8),
            "target_weight": labels["target_weight"].astype(np.int16),
            "bp_residual_weight": labels["bp_residual_weight"].astype(np.int16),
            "transport_k": np.array(int(getattr(code, "transport_k", code.k)), dtype=np.int16),
        }
        return row


def _worker_generate_chunk(cfg_json: str, split: str, target_count: int, seed: int, out_dir: str, shard_prefix: str) -> Dict[str, object]:
    # Keep workers CPU-only and avoid oversubscribing BLAS/OpenMP.
    # Slurm sets CUDA_VISIBLE_DEVICES in the parent job, so override it here rather
    # than using setdefault(). This prevents Sionna/TensorFlow worker processes from
    # grabbing the training GPU during dataset generation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import tensorflow as tf  # type: ignore
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
    except Exception:
        pass

    cfg = json.loads(cfg_json)
    code = build_code(cfg["code"])
    bp = BeliefPropagationDecoder(
        code,
        max_iters=int(cfg["data"].get("bp_collect_iterations", cfg["bp"].get("hybrid_main_iterations", 20))),
        algorithm=str(cfg["bp"].get("hybrid_main_algorithm", "nms")),
        nms_alpha=float(cfg["bp"].get("nms_alpha", 0.8)),
        early_stop=bool(cfg["bp"].get("early_stop", True)),
    )
    rng = np.random.default_rng(int(seed))
    shard_size = int(cfg["data"].get("shard_size", 250))
    rows: List[Dict[str, np.ndarray]] = []
    attempted = 0
    kept = 0
    shard_idx = 0
    out_dir_p = Path(out_dir)
    while kept < int(target_count):
        attempted += 1
        row = _make_row(cfg, code, bp, rng)
        rows.append(row)
        kept += 1
        if len(rows) >= shard_size:
            path = out_dir_p / f"{split}_{shard_prefix}_{shard_idx:04d}.npz"
            _flush_shard(path, rows)
            rows = []
            shard_idx += 1
    if rows:
        path = out_dir_p / f"{split}_{shard_prefix}_{shard_idx:04d}.npz"
        _flush_shard(path, rows)
    return {"split": split, "kept": kept, "attempted": attempted, "shards": shard_idx + (1 if rows else 0), "prefix": shard_prefix}


def _serial_generate(cfg: Dict[str, object], out_dir: Path) -> Dict[str, object]:
    logger = get_logger("generate")
    data_dir = ensure_dir(out_dir / "datasets")
    train_dir = ensure_dir(data_dir / "train")
    val_dir = ensure_dir(data_dir / "val")
    code = build_code(cfg["code"])
    rng = np.random.default_rng(int(cfg["project"].get("seed", 1234)))
    bp = BeliefPropagationDecoder(
        code,
        max_iters=int(cfg["data"].get("bp_collect_iterations", cfg["bp"].get("hybrid_main_iterations", 20))),
        algorithm=str(cfg["bp"].get("hybrid_main_algorithm", "nms")),
        nms_alpha=float(cfg["bp"].get("nms_alpha", 0.8)),
        early_stop=bool(cfg["bp"].get("early_stop", True)),
    )
    split_targets = {"train": int(cfg["data"]["train_samples"]), "val": int(cfg["data"]["val_samples"])}
    split_dirs = {"train": train_dir, "val": val_dir}
    shard_size = int(cfg["data"].get("shard_size", 250))
    kept = {"train": 0, "val": 0}
    attempted = 0
    buffers = {"train": [], "val": []}
    shard_idx = {"train": 0, "val": 0}

    def save_if_needed(split: str, force: bool = False):
        rows = buffers[split]
        if rows and (force or len(rows) >= shard_size):
            path = split_dirs[split] / f"{split}_shard_{shard_idx[split]:04d}.npz"
            _flush_shard(path, rows)
            logger.info("Wrote %s rows=%d", path, len(rows))
            buffers[split] = []
            shard_idx[split] += 1

    while kept["train"] < split_targets["train"] or kept["val"] < split_targets["val"]:
        attempted += 1
        row = _make_row(cfg, code, bp, rng)
        split = "train" if kept["train"] < split_targets["train"] else "val"
        buffers[split].append(row)
        kept[split] += 1
        save_if_needed(split)
        if attempted % 1000 == 0:
            logger.info("attempted=%d kept_train=%d kept_val=%d", attempted, kept["train"], kept["val"])

    save_if_needed("train", force=True)
    save_if_needed("val", force=True)
    return {"attempted": attempted, "kept": kept}


def generate_dataset(cfg: Dict[str, object]) -> None:
    logger = get_logger("generate")
    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]))
    data_dir = ensure_dir(out_dir / "datasets")
    train_dir = ensure_dir(data_dir / "train")
    val_dir = ensure_dir(data_dir / "val")
    artifacts = ensure_dir(out_dir / "artifacts")
    write_json(cfg, artifacts / "resolved_config.json")

    code = build_code(cfg["code"])
    code_summary = {
        "family": code.family,
        "requested_n": int(cfg["code"].get("n", cfg["code"].get("num_coded_bits", code.transmitted_n))),
        "requested_k": int(cfg["code"].get("k", getattr(code, "transport_k", code.k))),
        "transport_k": int(getattr(code, "transport_k", code.k)),
        "ldpc_k": int(code.k),
        "n_internal": code.n,
        "n_transmitted": code.transmitted_n,
        "k": code.k,
        "m": code.m,
        "rate": code.rate,
        "edges": int(code.h.sum()),
        "avg_check_degree": float(code.deg_c.mean()),
        "max_check_degree": int(code.deg_c.max()),
        "min_check_degree": int(code.deg_c.min()),
        "avg_variable_degree": float(code.deg_v.mean()),
        "num_punctured_internal_positions": int(code.punctured_positions.size),
        "num_transmitted_internal_positions": int(code.tx_positions.size),
        "bg": code.bg,
        "metadata": code.metadata,
    }
    write_json(code_summary, artifacts / "code_summary.json")
    logger.info("Code summary: %s", json.dumps(code_summary))

    num_workers = int(cfg["data"].get("num_workers", max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    num_workers = max(1, num_workers)
    if num_workers <= 1:
        summary = _serial_generate(cfg, out_dir)
        write_json(summary, data_dir / "generation_summary.json")
        logger.info("Serial generation complete: %s", summary)
        return

    logger.info("Parallel generation with %d workers", num_workers)
    for d in [train_dir, val_dir]:
        ensure_dir(d)
    split_targets = {"train": int(cfg["data"]["train_samples"]), "val": int(cfg["data"]["val_samples"])}
    chunk = int(cfg["data"].get("parallel_chunk_size", max(250, cfg["data"].get("shard_size", 250))))
    base_seed = int(cfg["project"].get("seed", 1234))
    jobs: List[Tuple[str, int, int, str, str]] = []
    job_idx = 0
    for split, target in split_targets.items():
        remaining = target
        while remaining > 0:
            take = min(chunk, remaining)
            out_subdir = str(train_dir if split == "train" else val_dir)
            jobs.append((split, take, base_seed + 100003 * (job_idx + 1), out_subdir, f"w{job_idx:03d}"))
            remaining -= take
            job_idx += 1

    summaries = []
    cfg_json = json.dumps(cfg)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker_generate_chunk, cfg_json, split, target, seed, out_subdir, prefix) for split, target, seed, out_subdir, prefix in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            summaries.append(res)
            logger.info("Completed worker chunk: %s", res)

    attempted = sum(int(x["attempted"]) for x in summaries)
    kept = {
        "train": sum(int(x["kept"]) for x in summaries if x["split"] == "train"),
        "val": sum(int(x["kept"]) for x in summaries if x["split"] == "val"),
    }
    write_json({"attempted": attempted, "kept": kept, "workers": num_workers, "jobs": summaries}, data_dir / "generation_summary.json")
    logger.info("Parallel generation complete: attempted=%d kept=%s", attempted, kept)
