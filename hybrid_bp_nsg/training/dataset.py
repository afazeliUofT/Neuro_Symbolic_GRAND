from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import math
import os
import time
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import Dataset

from ..codes.factory import build_code, code_summary
from ..channels.simulator import ChannelSimulationContext
from ..decoders.bp import BeliefPropagationDecoder
from .features import build_rescue_features, build_training_labels, PROFILE_TO_ID
from ..utils.io import ensure_dir, save_json


@dataclass
class SplitSpec:
    name: str
    num_samples: int
    num_shards: int


def _valid_npz(path: Path, min_records: int = 1) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "var_features" not in data.files:
                return False
            return int(data["var_features"].shape[0]) >= min_records
    except Exception:
        return False


def _sample_snr(rng: np.random.Generator, cfg: Dict[str, object], size: int) -> np.ndarray:
    data_cfg = cfg["data"]
    # Optional failure-biased distribution. This is important for failed-only rescue training,
    # because high-SNR NMS failures can be very rare and can stall generation.
    grid = data_cfg.get("failed_snr_db_grid") if data_cfg.get("sample_failed_only", False) else None
    probs = data_cfg.get("failed_snr_probs") if data_cfg.get("sample_failed_only", False) else None
    if grid is not None:
        grid_arr = np.asarray(grid, dtype=np.float32)
        if probs is None:
            p = np.ones(len(grid_arr), dtype=np.float64) / max(1, len(grid_arr))
        else:
            p = np.asarray(probs, dtype=np.float64)
            p = p / p.sum()
        return rng.choice(grid_arr, size=size, replace=True, p=p).astype(np.float32)
    return rng.uniform(float(data_cfg["snr_min_db"]), float(data_cfg["snr_max_db"]), size=size).astype(np.float32)


def _flush_npz_atomic(out_path: Path, records: Dict[str, List[np.ndarray]]) -> None:
    arrays = {k: np.stack(v, axis=0) for k, v in records.items()}
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, out_path)


def _generate_shard(cfg: Dict[str, object], split_name: str, shard_idx: int, shard_size: int, out_path: str) -> str:
    out_path_p = Path(out_path)
    if _valid_npz(out_path_p, min_records=max(1, int(cfg["data"].get("min_records_per_shard", 1)))):
        return f"SKIP {out_path}"

    project_seed = int(cfg["project"]["seed"])
    seed = project_seed + 1000 * (1 + shard_idx) + (0 if split_name == "train" else 100000)
    rng = np.random.default_rng(seed)
    code = build_code(cfg["code"])
    sim = ChannelSimulationContext(code=code, channel_cfg=cfg["channel"], seed=seed, tf_threads=1)
    bp_cfg = cfg["bp"]
    bp_decoder = BeliefPropagationDecoder(
        code=code,
        algorithm=str(bp_cfg.get("hybrid_main_algorithm", bp_cfg.get("algorithm", "spa"))),
        max_iters=int(cfg["data"].get("bp_collect_iterations", bp_cfg.get("hybrid_main_iterations", bp_cfg["main_iterations"]))),
        nms_alpha=float(bp_cfg.get("nms_alpha", 0.8)),
        llr_clip=float(bp_cfg.get("llr_clip", 18.0)),
        early_stop=bool(bp_cfg.get("early_stop", True)),
    )
    profiles = cfg["channel"]["train_profiles"] if split_name == "train" else cfg["channel"]["eval_profiles"]
    batch_size = int(cfg["data"].get("generation_batch_size", 64))
    max_attempts = int(cfg["data"].get("max_attempts_per_shard", 0))
    flush_every = int(cfg["data"].get("partial_flush_every", 0))

    records: Dict[str, List[np.ndarray]] = {
        "var_features": [],
        "check_features": [],
        "global_features": [],
        "heuristic_order": [],
        "bit_labels": [],
        "segment_labels": [],
        "weight_label": [],
        "standard_reachable": [],
        "expanded_reachable": [],
        "rescueable": [],
        "candidate_features": [],
        "candidate_labels": [],
        "candidate_valid": [],
        "profile_id": [],
        "snr_db": [],
        "bp_success": [],
    }
    kept = 0
    attempts = 0
    t0 = time.time()
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path_p.with_suffix(".progress.json")

    while kept < shard_size:
        if max_attempts and attempts >= max_attempts:
            break
        # Oversample when failed-only is enabled; most packets can be discarded.
        cur_batch = min(batch_size, max(shard_size - kept, 1) if not cfg["data"].get("sample_failed_only", False) else batch_size)
        prof_batch = [str(rng.choice(profiles)) for _ in range(cur_batch)]
        snr_batch = _sample_snr(rng, cfg, cur_batch)
        sim_batch = sim.simulate_batch(cur_batch, prof_batch, snr_batch)
        attempts += cur_batch
        for i in range(cur_batch):
            llr = sim_batch["llr"][i]
            true_code = sim_batch["codewords"][i]
            bp_result = bp_decoder.decode(llr, collect_trace=True)
            if bool(cfg["data"].get("sample_failed_only", False)) and bp_result.success:
                continue
            feat = build_rescue_features(code, llr, bp_result, float(snr_batch[i]), prof_batch[i], num_segments=int(cfg["model"]["num_segments"]))
            labels = build_training_labels(code, feat, true_code, bp_result, cfg["rescue"], cfg["model"])
            records["var_features"].append(feat["var_features"].astype(np.float16))
            records["check_features"].append(feat["check_features"].astype(np.float16))
            records["global_features"].append(feat["global_features"].astype(np.float16))
            records["heuristic_order"].append(feat["heuristic_order"].astype(np.int16))
            records["bit_labels"].append(labels["bit_labels"].astype(np.uint8))
            records["segment_labels"].append(labels["segment_labels"].astype(np.uint8))
            records["weight_label"].append(labels["weight_label"].astype(np.int16))
            records["standard_reachable"].append(labels["standard_reachable"].astype(np.uint8))
            records["expanded_reachable"].append(labels["expanded_reachable"].astype(np.uint8))
            records["rescueable"].append(labels["rescueable"].astype(np.uint8))
            records["candidate_features"].append(labels["candidate_features"].astype(np.float16))
            records["candidate_labels"].append(labels["candidate_labels"].astype(np.uint8))
            records["candidate_valid"].append(labels["candidate_valid"].astype(np.uint8))
            records["profile_id"].append(np.array(PROFILE_TO_ID.get(prof_batch[i], 0), dtype=np.int16))
            records["snr_db"].append(np.array(float(snr_batch[i]), dtype=np.float16))
            records["bp_success"].append(np.array(int(bp_result.success), dtype=np.uint8))
            kept += 1
            if flush_every and kept % flush_every == 0:
                # This is a safe final shard candidate; if the job is killed later,
                # the next run can skip this shard if it is sufficiently complete.
                _flush_npz_atomic(out_path_p, records)
            if kept >= shard_size:
                break
        if attempts % max(batch_size * 10, 1) == 0:
            save_json({"kept": kept, "target": shard_size, "attempts": attempts, "elapsed_s": time.time() - t0}, progress_path)

    if kept <= 0:
        raise RuntimeError(f"Shard {out_path} collected zero records after {attempts} attempts")
    _flush_npz_atomic(out_path_p, records)
    save_json({"kept": kept, "target": shard_size, "attempts": attempts, "elapsed_s": time.time() - t0, "complete": kept >= shard_size}, progress_path)
    return f"DONE {out_path} kept={kept} attempts={attempts}"


def _expected_shard_paths(output_dir: Path, split_name: str, num_shards: int) -> List[Path]:
    root = output_dir / "datasets" / split_name
    return [root / f"{split_name}_shard_{i:04d}.npz" for i in range(num_shards)]


def split_complete(output_dir: Path, split_name: str, num_shards: int) -> bool:
    return all(_valid_npz(p) for p in _expected_shard_paths(output_dir, split_name, num_shards))


def generate_supervised_dataset(cfg: Dict[str, object], output_dir: Path, logger) -> None:
    dataset_root = ensure_dir(output_dir / "datasets")
    try:
        code = build_code(cfg["code"])
        summary = code_summary(code)
        save_json(summary, output_dir / "artifacts" / "code_summary.json")
        save_json({"ok": True, "summary": summary}, output_dir / "artifacts" / "preflight_ok.json")
    except Exception:
        import traceback
        (output_dir / "artifacts" / "preflight_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("Code/channel preflight failed before dataset generation")
        raise

    split_specs = [
        SplitSpec("train", int(cfg["data"]["train_samples"]), int(cfg["data"]["train_shards"])),
        SplitSpec("val", int(cfg["data"]["val_samples"]), int(cfg["data"]["val_shards"])),
    ]
    for spec in split_specs:
        split_root = ensure_dir(dataset_root / spec.name)
        shard_size = int(math.ceil(spec.num_samples / spec.num_shards))
        pending = []
        for shard_idx in range(spec.num_shards):
            out_path = split_root / f"{spec.name}_shard_{shard_idx:04d}.npz"
            if _valid_npz(out_path):
                logger.info("Skipping existing %s", out_path)
            else:
                pending.append((shard_idx, out_path))
        if not pending:
            logger.info("%s split already complete with %d shards", spec.name, spec.num_shards)
            continue
        logger.info(
            "Generating %s split: %d/%d shards pending, shard_size=%d, workers=%d",
            spec.name,
            len(pending),
            spec.num_shards,
            shard_size,
            int(cfg["data"]["workers"]),
        )
        workers = int(cfg["data"].get("workers", 1))
        if workers <= 1:
            logger.info("Running %s generation serially to avoid multiprocessing/Sionna fork hazards", spec.name)
            for shard_idx, out_path in pending:
                msg = _generate_shard(cfg, spec.name, shard_idx, shard_size, str(out_path))
                logger.info("%s", msg)
        else:
            # TensorFlow/Sionna can hang when forked after module import on some CPU clusters.
            # Use spawn by default and recycle each child after a small number of shard tasks.
            start_method = str(cfg["data"].get("mp_start_method", "spawn"))
            max_tasks_per_child = int(cfg["data"].get("max_tasks_per_child", 1))
            heartbeat_s = float(cfg["data"].get("generation_heartbeat_s", 120.0))
            ctx = mp.get_context(start_method)
            ex_kwargs = {"max_workers": workers, "mp_context": ctx}
            if max_tasks_per_child > 0:
                ex_kwargs["max_tasks_per_child"] = max_tasks_per_child
            logger.info(
                "ProcessPool start_method=%s max_tasks_per_child=%s heartbeat_s=%.1f",
                start_method,
                max_tasks_per_child,
                heartbeat_s,
            )
            with cf.ProcessPoolExecutor(**ex_kwargs) as ex:
                futures = [ex.submit(_generate_shard, cfg, spec.name, shard_idx, shard_size, str(out_path)) for shard_idx, out_path in pending]
                future_set = set(futures)
                completed = 0
                while future_set:
                    done, future_set = cf.wait(future_set, timeout=heartbeat_s, return_when=cf.FIRST_COMPLETED)
                    if not done:
                        existing = sum(1 for pp in _expected_shard_paths(output_dir, spec.name, spec.num_shards) if _valid_npz(pp))
                        logger.info(
                            "Waiting for %s shards: completed_this_run=%d existing_valid=%d still_pending=%d",
                            spec.name,
                            completed,
                            existing,
                            len(future_set),
                        )
                        continue
                    for fut in done:
                        msg = fut.result()
                        completed += 1
                        logger.info("%s", msg)


class SupervisedShardDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        root = Path(root) / "datasets" / split
        files = sorted(root.glob(f"{split}_shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"No shard files found in {root}")
        arrays = []
        for f in files:
            if not _valid_npz(f):
                continue
            with np.load(f, allow_pickle=False) as data:
                arrays.append({k: data[k] for k in data.files})
        if not arrays:
            raise FileNotFoundError(f"No valid shard files found in {root}")
        self.data: Dict[str, np.ndarray] = {}
        for key in arrays[0].keys():
            self.data[key] = np.concatenate([a[key] for a in arrays], axis=0)
        self.length = int(self.data["var_features"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "var_features": torch.tensor(self.data["var_features"][idx], dtype=torch.float32),
            "check_features": torch.tensor(self.data["check_features"][idx], dtype=torch.float32),
            "global_features": torch.tensor(self.data["global_features"][idx], dtype=torch.float32),
            "heuristic_order": torch.tensor(self.data["heuristic_order"][idx], dtype=torch.long),
            "bit_labels": torch.tensor(self.data["bit_labels"][idx], dtype=torch.float32),
            "segment_labels": torch.tensor(self.data["segment_labels"][idx], dtype=torch.float32),
            "weight_label": torch.tensor(int(self.data["weight_label"][idx]), dtype=torch.long),
            "standard_reachable": torch.tensor(float(self.data["standard_reachable"][idx]), dtype=torch.float32),
            "expanded_reachable": torch.tensor(float(self.data["expanded_reachable"][idx]), dtype=torch.float32),
            "rescueable": torch.tensor(float(self.data["rescueable"][idx]), dtype=torch.float32),
            "candidate_features": torch.tensor(self.data["candidate_features"][idx], dtype=torch.float32),
            "candidate_labels": torch.tensor(self.data["candidate_labels"][idx], dtype=torch.float32),
            "candidate_valid": torch.tensor(self.data["candidate_valid"][idx], dtype=torch.float32),
        }
