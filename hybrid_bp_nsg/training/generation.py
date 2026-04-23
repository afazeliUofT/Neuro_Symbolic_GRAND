from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..channels.simulator import simulate_frame
from ..codes.factory import build_code
from ..decoders.bp import BeliefPropagationDecoder
from ..training.features import build_rescue_features, build_training_labels
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from ..config import normalize_snr_sampling


def _choose_snr_profile(cfg: Dict[str, object], rng: np.random.Generator):
    # Defensive normalization: load_config() already calls this, but this keeps
    # direct programmatic calls safe and prevents NumPy's `a and p must have same
    # size` failure if a shortened smoke/tail grid is paired with a full prior.
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
        "requested_n": int(cfg["code"].get("n", code.transmitted_n)),
        "requested_k": int(cfg["code"].get("k", code.k)),
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
            if path.exists():
                shard_idx[split] += 1
                path = split_dirs[split] / f"{split}_shard_{shard_idx[split]:04d}.npz"
            _flush_shard(path, rows)
            logger.info("Wrote %s rows=%d", path, len(rows))
            buffers[split] = []
            shard_idx[split] += 1

    sample_failed_only = bool(cfg["data"].get("sample_failed_only", True))
    while kept["train"] < split_targets["train"] or kept["val"] < split_targets["val"]:
        attempted += 1
        snr, profile = _choose_snr_profile(cfg, rng)
        frame = simulate_frame(code, snr, profile, rng)
        bp_result = bp.decode(frame.llr_internal, collect_trace=True)
        if sample_failed_only and bp_result.success:
            if attempted % 5000 == 0:
                logger.info("attempted=%d kept_train=%d kept_val=%d", attempted, kept["train"], kept["val"])
            continue

        fp = build_rescue_features(
            code, frame.llr_internal, bp_result, snr, profile,
            num_segments=int(cfg["model"]["num_segments"]),
            target_basis=str(cfg["rescue"].get("target_basis", "channel_with_bp_punctures")),
        )
        labels = build_training_labels(code, fp, frame.codeword_internal, bp_result, cfg["rescue"], cfg["model"])
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
            "profile_id": np.array({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(profile, 0), dtype=np.int16),
            "snr_db": np.array(snr, dtype=np.float32),
            "bp_success": np.array(int(bp_result.success), dtype=np.uint8),
            "target_weight": labels["target_weight"].astype(np.int16),
            "bp_residual_weight": labels["bp_residual_weight"].astype(np.int16),
        }
        split = "train" if kept["train"] < split_targets["train"] else "val"
        buffers[split].append(row)
        kept[split] += 1
        save_if_needed(split)
        if attempted % 1000 == 0:
            logger.info("attempted=%d kept_train=%d kept_val=%d", attempted, kept["train"], kept["val"])

    save_if_needed("train", force=True)
    save_if_needed("val", force=True)
    write_json({"attempted": attempted, "kept": kept}, data_dir / "generation_summary.json")
    logger.info("Generation complete: attempted=%d kept=%s", attempted, kept)
