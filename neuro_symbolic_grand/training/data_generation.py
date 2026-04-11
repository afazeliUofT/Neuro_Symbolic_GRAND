from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..codes import SystematicSparseCode, build_systematic_sparse_code
from ..config import ExperimentConfig
from ..utils.env import configure_process_thread_env, set_global_seed, write_runtime_snapshot
from ..utils.io import ensure_dir, read_json, write_json


def _code_from_artifact_dict(raw: Dict[str, object]) -> SystematicSparseCode:
    return SystematicSparseCode(
        n=int(raw["n"]),
        k=int(raw["k"]),
        p_matrix=np.asarray(raw["p_matrix"], dtype=np.uint8),
        seed=int(raw.get("seed", 0)),
    )


def prepare_code_artifact(cfg: ExperimentConfig, output_dir: Path) -> Path:
    code = build_systematic_sparse_code(
        n=cfg.code.n,
        k=cfg.code.k,
        p_column_weight=cfg.code.p_column_weight,
        seed=cfg.code.seed,
    )
    artifact_path = output_dir / "artifacts" / "code_artifact.json"
    code.save(artifact_path)
    return artifact_path


def _generate_shard_worker(payload: Dict[str, object]) -> Dict[str, object]:
    configure_process_thread_env(int(payload["tf_threads"]))
    from ..channels.simulator import ChannelSimulationContext

    shard_path = Path(str(payload["shard_path"]))
    summary_path = Path(str(payload["summary_path"]))
    code = _code_from_artifact_dict(payload["code_artifact"])
    rng = np.random.default_rng(int(payload["seed"]))
    profile_list = list(payload["profiles"])
    profile_to_id = {name: idx for idx, name in enumerate(profile_list)}
    sim = ChannelSimulationContext(
        code=code,
        channel_cfg=dict(payload["channel_cfg"]),
        seed=int(payload["seed"]),
        tf_threads=int(payload["tf_threads"]),
    )

    arrays: Dict[str, List[np.ndarray]] = {
        "messages": [],
        "codewords": [],
        "hard_bits": [],
        "error_mask": [],
        "llr": [],
        "snr_db": [],
        "h_freq_real": [],
        "h_freq_imag": [],
        "syndrome": [],
        "profile_id": [],
    }

    remaining = int(payload["num_samples"])
    batch_size = int(payload["batch_size"])
    while remaining > 0:
        current_bs = min(batch_size, remaining)
        profiles = rng.choice(profile_list, size=current_bs, replace=True).tolist()
        snr_db = rng.uniform(float(payload["snr_min_db"]), float(payload["snr_max_db"]), size=current_bs).astype(np.float32)
        batch = sim.simulate_batch(batch_size=current_bs, profiles=profiles, snr_db_values=snr_db)
        for key in ("messages", "codewords", "hard_bits", "error_mask", "llr", "snr_db", "h_freq_real", "h_freq_imag", "syndrome"):
            arrays[key].append(batch[key])
        arrays["profile_id"].append(np.array([profile_to_id[p] for p in profiles], dtype=np.int64))
        remaining -= current_bs

    final = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard_path, **final)
    summary = {
        "shard_path": str(shard_path),
        "summary_path": str(summary_path),
        "num_samples": int(final["llr"].shape[0]),
        "profiles": profile_list,
        "snr_min_db": float(final["snr_db"].min()),
        "snr_max_db": float(final["snr_db"].max()),
        "seed": int(payload["seed"]),
        "backends": {profile: type(backend).__name__ for profile, backend in sim.backends.items()},
    }
    write_json(summary, summary_path)
    return summary


def generate_train_val_datasets(cfg: ExperimentConfig, output_dir: Path, logger) -> Dict[str, object]:
    set_global_seed(cfg.seed)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "datasets")
    code_artifact_path = prepare_code_artifact(cfg, output_dir)
    code_artifact = read_json(code_artifact_path)
    write_runtime_snapshot(output_dir / "artifacts" / "runtime_snapshot.json")

    manifest: Dict[str, object] = {
        "experiment_name": cfg.experiment_name,
        "code_artifact": str(code_artifact_path),
        "profiles": cfg.channel.train_profiles,
        "profile_to_id": {name: idx for idx, name in enumerate(cfg.channel.train_profiles)},
        "splits": {},
    }

    for split_name, num_samples in (("train", cfg.data.train_samples), ("val", cfg.data.val_samples)):
        split_dir = ensure_dir(output_dir / "datasets" / split_name)
        summaries_dir = ensure_dir(split_dir / "summaries")
        num_shards = math.ceil(num_samples / cfg.data.shard_size)
        worker_count = max(1, int(cfg.resources.generation_workers))
        tasks = []
        for shard_id in range(num_shards):
            shard_samples = min(cfg.data.shard_size, num_samples - shard_id * cfg.data.shard_size)
            payload = {
                "split_name": split_name,
                "shard_id": shard_id,
                "num_samples": shard_samples,
                "batch_size": cfg.data.batch_size,
                "profiles": cfg.channel.train_profiles,
                "snr_min_db": cfg.data.snr_min_db,
                "snr_max_db": cfg.data.snr_max_db,
                "channel_cfg": cfg.channel.__dict__,
                "code_artifact": code_artifact,
                "seed": int(cfg.seed + 1000 * (1 if split_name == "val" else 0) + shard_id),
                "tf_threads": int(cfg.resources.generation_threads_per_worker),
                "shard_path": str(split_dir / f"{split_name}_shard_{shard_id:04d}.npz"),
                "summary_path": str(summaries_dir / f"{split_name}_shard_{shard_id:04d}.json"),
            }
            tasks.append(payload)

        logger.info("Generating %s split with %d shards using %d workers", split_name, num_shards, worker_count)
        shard_summaries: List[Dict[str, object]] = []
        if worker_count == 1:
            for payload in tasks:
                summary = _generate_shard_worker(payload)
                shard_summaries.append(summary)
                logger.info("Completed %s", summary["shard_path"])
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
            ) as ex:
                future_to_payload = {ex.submit(_generate_shard_worker, payload): payload for payload in tasks}
                for future in as_completed(future_to_payload):
                    summary = future.result()
                    shard_summaries.append(summary)
                    logger.info("Completed %s", summary["shard_path"])

        shard_summaries.sort(key=lambda x: x["shard_path"])
        split_manifest = {
            "num_samples": num_samples,
            "num_shards": num_shards,
            "shards": shard_summaries,
        }
        write_json(split_manifest, split_dir / f"{split_name}_manifest.json")
        manifest["splits"][split_name] = split_manifest

    write_json(manifest, output_dir / "datasets" / "dataset_manifest.json")
    return manifest


def load_dataset_arrays(split_dir: Path) -> Dict[str, np.ndarray]:
    manifest_name = f"{split_dir.name}_manifest.json"
    split_manifest = read_json(split_dir / manifest_name)
    arrays: Dict[str, List[np.ndarray]] = {}
    for shard in split_manifest["shards"]:
        with np.load(shard["shard_path"], allow_pickle=False) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
