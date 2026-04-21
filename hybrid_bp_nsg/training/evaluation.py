from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Dict, List
import gzip
import os
import numpy as np
import pandas as pd
from scipy.stats import beta

from ..codes.factory import build_code
from ..channels.simulator import ChannelSimulationContext
from ..decoders.hybrid import HybridDecoderFactory
from ..utils.io import ensure_dir


def _decoder_list(cfg: Dict[str, object]) -> List[str]:
    names = []
    if cfg["benchmarks"].get("evaluate_bp", True):
        names += [f"bp_{it}" for it in cfg["benchmarks"].get("bp_iteration_list", [10, 20, 50])]
    if cfg["benchmarks"].get("evaluate_nms", True):
        names += [f"bp_nms_{it}" for it in cfg["benchmarks"].get("nms_iteration_list", [20, 50])]
    if cfg["benchmarks"].get("evaluate_wbf_post", True):
        names += ["bp_wbf_post"]
    if cfg["benchmarks"].get("evaluate_orb_rescue", True):
        names += ["bp_orb_rescue"]
    if cfg["benchmarks"].get("evaluate_cdf_rescue", True):
        names += ["bp_cdf_rescue"]
    if cfg["benchmarks"].get("evaluate_segmented_rescue", True):
        names += ["bp_segmented_rescue"]
    if cfg["benchmarks"].get("evaluate_hybrid", True):
        names += ["hybrid_bp_nsg"]
    return names


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k <= 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def _max_samples(eval_cfg: Dict[str, object], tail: bool) -> int:
    if tail:
        return int(eval_cfg.get("max_samples_per_point_tail", eval_cfg.get("tail_samples_per_point", 30000)))
    return int(eval_cfg.get("max_samples_per_point", eval_cfg.get("samples_per_point", 4000)))


def _target_errors(eval_cfg: Dict[str, object], tail: bool) -> int:
    if tail:
        return int(eval_cfg.get("target_frame_errors_tail", eval_cfg.get("target_frame_errors", 200)))
    return int(eval_cfg.get("target_frame_errors", 200))


def _evaluate_point_worker(cfg: Dict[str, object], profile: str, snr_db: float, checkpoint_path: str | None,
                           point_root: str, tail: bool = False) -> Dict[str, object]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import torch
    torch.set_num_threads(int(cfg["eval"].get("torch_threads_per_worker", 1)))
    code = build_code(cfg["code"])
    sim = ChannelSimulationContext(
        code=code,
        channel_cfg=cfg["channel"],
        seed=int(cfg["project"]["seed"]) + int(1000 * snr_db) + ord(profile[0]),
        tf_threads=1,
    )
    factory = HybridDecoderFactory(code, cfg, checkpoint_path=checkpoint_path, device="cpu")
    decoders = _decoder_list(cfg)
    eval_cfg = cfg["eval"]
    stop_decoders = [d for d in eval_cfg.get("stop_decoders", ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"]) if d in decoders]
    max_samples = _max_samples(eval_cfg, tail)
    target_errors = _target_errors(eval_cfg, tail)
    min_errors_to_report = int(eval_cfg.get("min_frame_errors_to_report", 30))
    batch_size = int(eval_cfg.get("batch_size", 32))

    rows = []
    err_counts = {d: 0 for d in stop_decoders}
    sample_idx = 0
    while sample_idx < max_samples:
        cur = min(batch_size, max_samples - sample_idx)
        sim_batch = sim.simulate_batch(cur, [profile] * cur, np.full(cur, snr_db, dtype=np.float32))
        for i in range(cur):
            llr = sim_batch["llr"][i]
            true_codeword = sim_batch["codewords"][i]
            out = factory.evaluate_all(llr, true_codeword, snr_db, profile)
            for name in decoders:
                r = out[name]
                be = int(r.block_error)
                rows.append({
                    "profile": profile,
                    "snr_db": float(snr_db),
                    "sample_idx": sample_idx + i,
                    "decoder": name,
                    "success": int(r.success),
                    "block_error": be,
                    "latency_ms": float(r.latency_ms),
                    "queries": int(r.queries),
                    "action": r.action,
                    "rescue_used": int(r.rescue_used),
                    "micro_bp_used": int(r.micro_bp_used),
                    "main_success": int(r.main_success),
                })
                if name in err_counts:
                    err_counts[name] += be
        sample_idx += cur
        if stop_decoders and all(err_counts[d] >= target_errors for d in stop_decoders):
            break

    point_root = Path(point_root)
    ensure_dir(point_root)
    raw_path = point_root / "raw_records.csv.gz"
    df = pd.DataFrame(rows)
    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)

    summary_rows = []
    for decoder, group in df.groupby("decoder"):
        n = int(len(group))
        frame_errors = int(group["block_error"].sum())
        lo, hi = _clopper_pearson(frame_errors, n)
        summary_rows.append({
            "profile": profile,
            "snr_db": float(snr_db),
            "decoder": decoder,
            "samples": n,
            "frame_errors": frame_errors,
            "bler": float(group["block_error"].mean()),
            "ci_low": lo,
            "ci_high": hi,
            "plot_eligible": int(frame_errors >= min_errors_to_report),
            "avg_latency_ms": float(group["latency_ms"].mean()),
            "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
            "p99_latency_ms": float(group["latency_ms"].quantile(0.99)),
            "avg_queries": float(group["queries"].mean()),
            "p95_queries": float(group["queries"].quantile(0.95)),
            "rescue_rate": float(group["rescue_used"].mean()),
            "micro_bp_rate": float(group["micro_bp_used"].mean()),
            "main_success_rate": float(group["main_success"].mean()),
            "stop_target_errors": target_errors,
            "max_samples_cap": max_samples,
        })
    pd.DataFrame(summary_rows).to_csv(point_root / "summary.csv", index=False)
    return {
        "profile": profile,
        "snr_db": snr_db,
        "point_root": str(point_root),
        "summary_rows": summary_rows,
    }



def _point_complete(point_root: Path) -> bool:
    return (point_root / "summary.csv").exists() and (point_root / "raw_records.csv.gz").exists()


def evaluate_grid(cfg: Dict[str, object], output_dir: Path, logger, tail: bool = False) -> None:
    eval_cfg = cfg["eval"]
    profiles = list(eval_cfg["profiles"])
    snr_grid = list(eval_cfg["tail_snr_db_grid"] if tail else eval_cfg["snr_db_grid"])
    eval_root = ensure_dir(output_dir / "evaluation")
    final_summary = eval_root / "evaluation_summary.csv"
    final_raw = eval_root / "all_raw_records.csv.gz"
    if final_summary.exists() and final_raw.exists():
        logger.info("Evaluation already complete at %s; skipping evaluation stage", eval_root)
        return

    checkpoint_path = output_dir / "checkpoints" / "rescue_net.pt"
    ckpt = str(checkpoint_path) if checkpoint_path.exists() else None
    ops = [(p, float(s), eval_root / f"profile_{p}" / f"snr_{s:+.1f}dB") for p in profiles for s in snr_grid]

    pending = []
    for p, s, point_root in ops:
        if _point_complete(point_root):
            logger.info("Skipping completed profile=%s snr=%.1f dB", p, s)
        else:
            pending.append((p, s, point_root))

    logger.info("Evaluating %d/%d pending operating points", len(pending), len(ops))
    if pending:
        futures = []
        with cf.ProcessPoolExecutor(max_workers=int(eval_cfg["workers"])) as ex:
            for p, s, point_root in pending:
                futures.append(ex.submit(_evaluate_point_worker, cfg, p, s, ckpt, str(point_root), tail))
            for fut in cf.as_completed(futures):
                result = fut.result()
                logger.info("Completed profile=%s snr=%.1f dB", result["profile"], result["snr_db"])

    missing = [str(point_root) for _, _, point_root in ops if not _point_complete(point_root)]
    if missing:
        raise RuntimeError(f"Evaluation did not complete all operating points. Missing: {missing[:10]}")

    summary_parts = [pd.read_csv(point_root / "summary.csv") for _, _, point_root in ops]
    summary_df = pd.concat(summary_parts, axis=0, ignore_index=True).sort_values(["profile", "snr_db", "decoder"]).reset_index(drop=True)
    summary_df.to_csv(final_summary, index=False)

    raw_parts = [pd.read_csv(point_root / "raw_records.csv.gz") for _, _, point_root in ops]
    raw_df = pd.concat(raw_parts, axis=0, ignore_index=True)
    with gzip.open(final_raw, "wt", encoding="utf-8") as f:
        raw_df.to_csv(f, index=False)
