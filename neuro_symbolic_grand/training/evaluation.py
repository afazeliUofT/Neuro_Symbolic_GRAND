from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..codes import SystematicSparseCode, load_code_artifact
from ..config import ExperimentConfig
from ..utils.env import configure_process_thread_env, configure_torch_threads, set_global_seed
from ..utils.io import append_jsonl, ensure_dir, read_json, write_dataframe_csv, write_json


def _code_from_artifact_dict(raw: Dict[str, object]) -> SystematicSparseCode:
    return SystematicSparseCode(
        n=int(raw["n"]),
        k=int(raw["k"]),
        p_matrix=np.asarray(raw["p_matrix"], dtype=np.uint8),
        seed=int(raw.get("seed", 0)),
    )


def _flatten_trace(trace) -> List[Dict[str, object]]:
    out = []
    for attempt in trace:
        out.append(
            {
                "stage": attempt.stage,
                "query_index": attempt.query_index,
                "weight": attempt.weight,
                "score": attempt.score,
                "positions": attempt.positions,
                "syndrome_weight": attempt.syndrome_weight,
                "success": attempt.success,
            }
        )
    return out


def _evaluate_point_worker(payload: Dict[str, object]) -> Dict[str, object]:
    configure_process_thread_env(int(payload["tf_threads"]))
    configure_torch_threads(1, 1)
    from ..channels.simulator import ChannelSimulationContext
    from ..decoders import NeuroSymbolicGRAND, WeightedReliabilityGRAND
    from ..training.trainer import load_trained_model

    point_dir = Path(str(payload["point_dir"]))
    traces_path = point_dir / "traces.jsonl"
    raw_csv_path = point_dir / "raw_records.csv"
    summary_path = point_dir / "summary.json"
    point_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig()
    # The caller already materializes the values we need in payload.
    code = _code_from_artifact_dict(payload["code_artifact"])
    profile_id = int(payload["profile_id"])
    profile_name = str(payload["profile_name"])
    snr_db = float(payload["snr_db"])
    samples_per_point = int(payload["samples_per_point"])
    batch_size = int(payload["batch_size"])
    rng = np.random.default_rng(int(payload["seed"]))

    model = load_trained_model(Path(str(payload["output_dir"])), device="cpu")
    baseline = WeightedReliabilityGRAND(
        code=code,
        pool_size=int(payload["baseline_pool_size"]),
        max_weight=int(payload["baseline_max_weight"]),
        budget=int(payload["baseline_budget"]),
        weight_penalties=list(payload["baseline_weight_penalties"]),
        trace_top_attempts=int(payload["trace_top_attempts"]),
    )
    ns_decoder = NeuroSymbolicGRAND(
        code=code,
        model=model,
        device="cpu",
        num_segments=int(payload["num_segments"]),
        max_weight_class=int(payload["max_weight_class"]),
        ai_pool_size=int(payload["ai_pool_size"]),
        ai_max_weight=int(payload["ai_max_weight"]),
        ai_budget=int(payload["ai_budget"]),
        fallback_decoder=WeightedReliabilityGRAND(
            code=code,
            pool_size=int(payload["baseline_pool_size"]),
            max_weight=int(payload["baseline_max_weight"]),
            budget=int(payload["fallback_budget"]),
            weight_penalties=list(payload["baseline_weight_penalties"]),
            trace_top_attempts=int(payload["trace_top_attempts"]),
        ),
        ai_weight_penalties=list(payload["ai_weight_penalties"]),
        top_segments=int(payload["top_segments"]),
        top_bits_extra=int(payload["top_bits_extra"]),
        confidence_threshold=float(payload["confidence_threshold"]),
        candidate_mass_threshold=float(payload["candidate_mass_threshold"]),
        trace_top_attempts=int(payload["trace_top_attempts"]),
    )
    sim = ChannelSimulationContext(
        code=code,
        channel_cfg=dict(payload["channel_cfg"]),
        seed=int(payload["seed"]),
        tf_threads=int(payload["tf_threads"]),
    )

    records: List[Dict[str, object]] = []
    trace_records: List[Dict[str, object]] = []
    processed = 0
    while processed < samples_per_point:
        current_bs = min(batch_size, samples_per_point - processed)
        batch = sim.simulate_batch(
            batch_size=current_bs,
            profiles=[profile_name] * current_bs,
            snr_db_values=np.full(current_bs, snr_db, dtype=np.float32),
        )
        for idx in range(current_bs):
            llr = batch["llr"][idx]
            hard_bits = batch["hard_bits"][idx]
            truth_codeword = batch["codewords"][idx]
            truth_error_mask = batch["error_mask"][idx]
            true_error_weight = int(truth_error_mask.sum())
            baseline_result = baseline.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
            ns_result = ns_decoder.decode(
                llr=llr,
                hard_bits=hard_bits,
                snr_db=snr_db,
                profile_id=profile_id,
                truth_error_mask=truth_error_mask,
            )
            for decoder_name, result in (("baseline", baseline_result), ("nsgrand", ns_result)):
                bit_errors = int(np.sum(result.decoded_codeword != truth_codeword))
                block_error = int(bit_errors > 0)
                records.append(
                    {
                        "profile": profile_name,
                        "profile_id": profile_id,
                        "snr_db": snr_db,
                        "decoder": decoder_name,
                        "true_error_weight": true_error_weight,
                        "block_error": block_error,
                        "bit_errors": bit_errors,
                        "queries": int(result.queries),
                        "elapsed_ms": float(result.elapsed_ms),
                        "fallback_used": int(bool(result.fallback_used)),
                        "candidate_pool_size": int(result.candidate_pool_size),
                        "oracle_pool_hit": int(result.oracle_pool_hit) if result.oracle_pool_hit is not None else -1,
                        "predicted_weight_top": int(result.predicted_weight_top) if result.predicted_weight_top is not None else -1,
                        "confidence_prob": float(result.confidence_prob) if result.confidence_prob is not None else np.nan,
                        "stage": result.stage,
                    }
                )
                if rng.random() < float(payload["trace_fraction"]):
                    trace_records.append(
                        {
                            "profile": profile_name,
                            "snr_db": snr_db,
                            "decoder": decoder_name,
                            "true_error_weight": true_error_weight,
                            "block_error": block_error,
                            "bit_errors": bit_errors,
                            "queries": int(result.queries),
                            "fallback_used": bool(result.fallback_used),
                            "confidence_prob": float(result.confidence_prob) if result.confidence_prob is not None else None,
                            "trace": _flatten_trace(result.trace),
                        }
                    )
        processed += current_bs

    raw_df = pd.DataFrame(records)
    write_dataframe_csv(raw_df, raw_csv_path)
    if trace_records:
        append_jsonl(trace_records, traces_path)

    summary_rows = []
    for decoder_name, sub_df in raw_df.groupby("decoder"):
        summary_rows.append(
            {
                "profile": profile_name,
                "profile_id": profile_id,
                "snr_db": snr_db,
                "decoder": decoder_name,
                "num_samples": int(len(sub_df)),
                "bler": float(sub_df["block_error"].mean()),
                "ber": float(sub_df["bit_errors"].sum() / (len(sub_df) * code.n)),
                "avg_queries": float(sub_df["queries"].mean()),
                "median_queries": float(sub_df["queries"].median()),
                "p95_queries": float(sub_df["queries"].quantile(0.95)),
                "p99_queries": float(sub_df["queries"].quantile(0.99)),
                "avg_elapsed_ms": float(sub_df["elapsed_ms"].mean()),
                "fallback_rate": float(sub_df["fallback_used"].mean()),
                "oracle_pool_hit_rate": float(sub_df.loc[sub_df["oracle_pool_hit"] >= 0, "oracle_pool_hit"].mean()) if (sub_df["oracle_pool_hit"] >= 0).any() else None,
            }
        )
    write_json({"rows": summary_rows}, summary_path)
    return {"summary_path": str(summary_path), "raw_csv_path": str(raw_csv_path), "rows": summary_rows}


def evaluate_grid(cfg: ExperimentConfig, output_dir: Path, logger) -> Dict[str, object]:
    set_global_seed(cfg.seed)
    code_artifact = read_json(output_dir / "artifacts" / "code_artifact.json")
    eval_root = ensure_dir(output_dir / "evaluation")
    tasks = []
    for profile_id, profile_name in enumerate(cfg.channel.eval_profiles):
        for snr_db in cfg.eval.snr_grid_db:
            point_dir = eval_root / f"profile_{profile_name}" / f"snr_{snr_db:+.1f}dB"
            tasks.append(
                {
                    "output_dir": str(output_dir),
                    "point_dir": str(point_dir),
                    "code_artifact": code_artifact,
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "snr_db": float(snr_db),
                    "samples_per_point": int(cfg.eval.samples_per_point),
                    "batch_size": int(cfg.eval.batch_size),
                    "trace_fraction": float(cfg.eval.trace_fraction),
                    "seed": int(cfg.seed + 200000 + profile_id * 1000 + int((snr_db + 20) * 10)),
                    "tf_threads": int(cfg.resources.evaluation_threads_per_worker),
                    "channel_cfg": cfg.channel.__dict__,
                    "baseline_pool_size": int(cfg.search.baseline_pool_size),
                    "baseline_max_weight": int(cfg.search.baseline_max_weight),
                    "baseline_budget": int(cfg.search.baseline_budget),
                    "baseline_weight_penalties": list(cfg.search.baseline_weight_penalties),
                    "num_segments": int(cfg.model.num_segments),
                    "max_weight_class": int(cfg.model.max_weight_class),
                    "ai_pool_size": int(cfg.search.ai_pool_size),
                    "ai_max_weight": int(cfg.search.ai_max_weight),
                    "ai_budget": int(cfg.search.ai_budget),
                    "fallback_budget": int(cfg.search.fallback_budget),
                    "ai_weight_penalties": list(cfg.search.ai_weight_penalties),
                    "top_segments": int(cfg.search.top_segments),
                    "top_bits_extra": int(cfg.search.top_bits_extra),
                    "confidence_threshold": float(cfg.search.confidence_threshold),
                    "candidate_mass_threshold": float(cfg.search.candidate_mass_threshold),
                    "trace_top_attempts": int(cfg.search.trace_top_attempts),
                }
            )

    logger.info("Evaluating %d operating points", len(tasks))
    all_rows: List[Dict[str, object]] = []
    worker_count = max(1, int(cfg.resources.evaluation_workers))
    if worker_count == 1:
        for payload in tasks:
            result = _evaluate_point_worker(payload)
            logger.info("Completed evaluation at %s", result["summary_path"])
            all_rows.extend(result["rows"])
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
        ) as ex:
            future_to_task = {ex.submit(_evaluate_point_worker, payload): payload for payload in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                logger.info("Completed evaluation at %s", result["summary_path"])
                all_rows.extend(result["rows"])

    summary_df = pd.DataFrame(all_rows).sort_values(["profile", "snr_db", "decoder"]).reset_index(drop=True)
    write_dataframe_csv(summary_df, eval_root / "evaluation_summary.csv")
    write_json({"rows": all_rows}, eval_root / "evaluation_summary.json")
    return {"num_rows": int(len(summary_df)), "summary_csv": str(eval_root / "evaluation_summary.csv")}
