from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..codes import SystematicSparseCode
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


def _jsonify(value: object) -> str:
    return json.dumps(value, sort_keys=True)


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


def _trace_category_for_ns(record: Dict[str, object]) -> Optional[str]:
    if str(record.get("decoder")) != "nsgrand":
        return None
    stage = str(record.get("stage"))
    gate_reason = str(record.get("gate_reason"))
    block_error = int(record.get("block_error", 1))
    queries = int(record.get("queries", 0))
    primary_budget = max(1, int(record.get("primary_budget", 1)))
    if gate_reason == "presearch_skip_hopeless":
        return "presearch_skip_hopeless"
    if stage == "ai" and block_error == 0:
        return "ai_success"
    if gate_reason.startswith("presearch") and block_error == 0:
        return "presearch_fallback_success"
    if gate_reason.startswith("presearch") and block_error == 1:
        return "presearch_fallback_fail"
    if gate_reason == "postsearch_exhausted" and block_error == 0:
        return "postsearch_fallback_success"
    if gate_reason == "postsearch_exhausted" and block_error == 1:
        return "postsearch_fallback_fail"
    if block_error == 1:
        return "final_failure"
    if queries >= int(0.8 * primary_budget):
        return "high_query_tail"
    return None


def _record_from_result(
    *,
    sample_id: int,
    profile_name: str,
    profile_id: int,
    snr_db: float,
    decoder_name: str,
    true_error_weight: int,
    truth_codeword: np.ndarray,
    result,
) -> Dict[str, object]:
    bit_errors = int(np.sum(result.decoded_codeword != truth_codeword))
    block_error = int(bit_errors > 0)
    diagnostics = result.diagnostics or {}
    return {
        "sample_id": int(sample_id),
        "profile": profile_name,
        "profile_id": int(profile_id),
        "snr_db": float(snr_db),
        "decoder": decoder_name,
        "true_error_weight": int(true_error_weight),
        "decoder_success": int(bool(result.success)),
        "block_error": int(block_error),
        "bit_errors": int(bit_errors),
        "queries": int(result.queries),
        "primary_queries": int(getattr(result, "primary_queries", result.queries)),
        "fallback_queries": int(getattr(result, "fallback_queries", 0)),
        "elapsed_ms": float(result.elapsed_ms),
        "primary_elapsed_ms": float(getattr(result, "primary_elapsed_ms", result.elapsed_ms)),
        "fallback_elapsed_ms": float(getattr(result, "fallback_elapsed_ms", 0.0)),
        "fallback_used": int(bool(result.fallback_used)),
        "candidate_pool_size": int(result.candidate_pool_size),
        "oracle_pool_hit": int(result.oracle_pool_hit) if result.oracle_pool_hit is not None else -1,
        "predicted_weight_top": int(result.predicted_weight_top) if result.predicted_weight_top is not None else -1,
        "confidence_prob": float(result.confidence_prob) if result.confidence_prob is not None else np.nan,
        "predicted_overflow_prob": float(getattr(result, "predicted_overflow_prob", np.nan)) if getattr(result, "predicted_overflow_prob", None) is not None else np.nan,
        "stage": str(result.stage),
        "gate_reason": str(getattr(result, "gate_reason", "none")),
        "search_mode": str(diagnostics.get("search_mode", "none")),
        "policy_action": str(diagnostics.get("policy_action", "none")),
        "allowed_weights_json": _jsonify(diagnostics.get("allowed_weights", [])),
        "top_segment_ids_json": _jsonify(diagnostics.get("top_segment_ids", [])),
        "pool_positions_top10_json": _jsonify(diagnostics.get("pool_positions_top10", [])),
        "predicted_weight_distribution_json": _jsonify(diagnostics.get("predicted_weight_distribution", [])),
        "primary_budget": int(diagnostics.get("primary_budget", 0)),
        "primary_max_weight": int(diagnostics.get("primary_max_weight", 0)),
        "fallback_budget": int(diagnostics.get("fallback_budget", 0)),
        "fallback_pool_size": int(diagnostics.get("fallback_pool_size", 0)),
        "fallback_max_weight": int(diagnostics.get("fallback_max_weight", 0)),
        "ai_success": int(bool(diagnostics.get("ai_success", False))),
        "fallback_success": int(bool(diagnostics.get("fallback_success", False))),
        "presearch_gate": int(bool(diagnostics.get("presearch_gate", False))),
    }


def _summarize_decoder_frame(sub_df: pd.DataFrame, *, profile_name: str, profile_id: int, snr_db: float, code_n: int) -> Dict[str, object]:
    row = {
        "profile": profile_name,
        "profile_id": int(profile_id),
        "snr_db": float(snr_db),
        "decoder": str(sub_df["decoder"].iloc[0]),
        "num_samples": int(len(sub_df)),
        "bler": float(sub_df["block_error"].mean()),
        "ber": float(sub_df["bit_errors"].sum() / max(1, len(sub_df) * code_n)),
        "avg_queries": float(sub_df["queries"].mean()),
        "avg_primary_queries": float(sub_df["primary_queries"].mean()),
        "avg_fallback_queries": float(sub_df["fallback_queries"].mean()),
        "median_queries": float(sub_df["queries"].median()),
        "p95_queries": float(sub_df["queries"].quantile(0.95)),
        "p99_queries": float(sub_df["queries"].quantile(0.99)),
        "avg_elapsed_ms": float(sub_df["elapsed_ms"].mean()),
        "avg_primary_elapsed_ms": float(sub_df["primary_elapsed_ms"].mean()),
        "avg_fallback_elapsed_ms": float(sub_df["fallback_elapsed_ms"].mean()),
        "p95_elapsed_ms": float(sub_df["elapsed_ms"].quantile(0.95)),
        "p99_elapsed_ms": float(sub_df["elapsed_ms"].quantile(0.99)),
        "fallback_rate": float(sub_df["fallback_used"].mean()),
        "oracle_pool_hit_rate": float(sub_df.loc[sub_df["oracle_pool_hit"] >= 0, "oracle_pool_hit"].mean()) if (sub_df["oracle_pool_hit"] >= 0).any() else None,
        "avg_confidence_prob": float(sub_df["confidence_prob"].dropna().mean()) if sub_df["confidence_prob"].notna().any() else None,
        "avg_predicted_overflow_prob": float(sub_df["predicted_overflow_prob"].dropna().mean()) if sub_df["predicted_overflow_prob"].notna().any() else None,
    }
    return row


def _build_pairwise_df(raw_df: pd.DataFrame, ref_decoder: str, target_decoder: str) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    ref = raw_df[raw_df["decoder"] == ref_decoder].copy()
    target = raw_df[raw_df["decoder"] == target_decoder].copy()
    if ref.empty or target.empty:
        return pd.DataFrame()
    ref = ref.add_prefix(f"{ref_decoder}_")
    target = target.add_prefix(f"{target_decoder}_")
    paired = ref.merge(
        target,
        left_on=[f"{ref_decoder}_sample_id", f"{ref_decoder}_profile", f"{ref_decoder}_snr_db"],
        right_on=[f"{target_decoder}_sample_id", f"{target_decoder}_profile", f"{target_decoder}_snr_db"],
        how="inner",
    )
    paired["sample_id"] = paired[f"{ref_decoder}_sample_id"]
    paired["profile"] = paired[f"{ref_decoder}_profile"]
    paired["snr_db"] = paired[f"{ref_decoder}_snr_db"]
    paired["true_error_weight"] = paired[f"{ref_decoder}_true_error_weight"]
    paired["reference_decoder"] = ref_decoder
    paired["target_decoder"] = target_decoder
    paired["query_delta"] = paired[f"{target_decoder}_queries"] - paired[f"{ref_decoder}_queries"]
    paired["latency_delta_ms"] = paired[f"{target_decoder}_elapsed_ms"] - paired[f"{ref_decoder}_elapsed_ms"]
    paired["block_error_delta"] = paired[f"{target_decoder}_block_error"] - paired[f"{ref_decoder}_block_error"]
    paired[f"{target_decoder}_better_block_error"] = ((paired[f"{target_decoder}_block_error"] < paired[f"{ref_decoder}_block_error"])).astype(int)
    paired[f"{ref_decoder}_better_block_error"] = ((paired[f"{ref_decoder}_block_error"] < paired[f"{target_decoder}_block_error"])).astype(int)
    return paired


def _build_paired_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return _build_pairwise_df(raw_df, ref_decoder="baseline", target_decoder="nsgrand")


def _evaluate_point_worker(payload: Dict[str, object]) -> Dict[str, object]:
    configure_process_thread_env(int(payload["tf_threads"]))
    configure_torch_threads(1, 1)
    from ..channels.simulator import ChannelSimulationContext
    from ..decoders import NeuroSymbolicGRAND, WeightedReliabilityGRAND
    from ..training.trainer import load_trained_model

    point_dir = Path(str(payload["point_dir"]))
    traces_path = point_dir / "traces.jsonl"
    interesting_traces_path = point_dir / "interesting_traces.jsonl"
    raw_csv_path = point_dir / "raw_records.csv"
    paired_csv_path = point_dir / "paired_records.csv"
    summary_path = point_dir / "summary.json"
    point_dir.mkdir(parents=True, exist_ok=True)

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
    strong_symbolic = None
    if bool(payload.get("evaluate_strong_symbolic", False)):
        strong_symbolic = WeightedReliabilityGRAND(
            code=code,
            pool_size=int(payload["strong_pool_size"]),
            max_weight=int(payload["strong_max_weight"]),
            budget=int(payload["strong_budget"]),
            weight_penalties=list(payload["strong_weight_penalties"]),
            trace_top_attempts=int(payload["trace_top_attempts"]),
        )

    ns_fallback_kind = str(payload.get("ns_fallback_kind", "strong")).strip().lower()
    if ns_fallback_kind == "baseline":
        ns_fallback_decoder = WeightedReliabilityGRAND(
            code=code,
            pool_size=int(payload["baseline_pool_size"]),
            max_weight=int(payload["baseline_max_weight"]),
            budget=int(payload["baseline_budget"]),
            weight_penalties=list(payload["baseline_weight_penalties"]),
            trace_top_attempts=int(payload["trace_top_attempts"]),
        )
    else:
        ns_fallback_decoder = WeightedReliabilityGRAND(
            code=code,
            pool_size=int(payload["fallback_pool_size"]),
            max_weight=int(payload["fallback_max_weight"]),
            budget=int(payload["fallback_budget"]),
            weight_penalties=list(payload["fallback_weight_penalties"]),
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
        fallback_decoder=ns_fallback_decoder,
        ai_weight_penalties=list(payload["ai_weight_penalties"]),
        top_segments=int(payload["top_segments"]),
        top_bits_extra=int(payload["top_bits_extra"]),
        confidence_threshold=float(payload["confidence_threshold"]),
        candidate_mass_threshold=float(payload["candidate_mass_threshold"]),
        adaptive_pool_size=int(payload["adaptive_pool_size"]),
        adaptive_max_weight=int(payload["adaptive_max_weight"]),
        adaptive_budget=int(payload["adaptive_budget"]),
        adaptive_top_segments=int(payload["adaptive_top_segments"]),
        adaptive_top_bits_extra=int(payload["adaptive_top_bits_extra"]),
        overflow_expand_threshold=float(payload["overflow_expand_threshold"]),
        overflow_direct_fallback_threshold=float(payload["overflow_direct_fallback_threshold"]),
        confidence_presearch_threshold=float(payload["confidence_presearch_threshold"]),
        overflow_direct_action=str(payload.get("overflow_direct_action", "fallback")),
        overflow_direct_confidence_ceiling=float(payload.get("overflow_direct_confidence_ceiling", 1.0)),
        always_fallback_after_ai_fail=bool(payload["always_fallback_after_ai_fail"]),
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
    interesting_trace_records: List[Dict[str, object]] = []
    interesting_counts: Dict[str, int] = {}
    processed = 0
    sample_offset = 0
    while processed < samples_per_point:
        current_bs = min(batch_size, samples_per_point - processed)
        batch = sim.simulate_batch(
            batch_size=current_bs,
            profiles=[profile_name] * current_bs,
            snr_db_values=np.full(current_bs, snr_db, dtype=np.float32),
        )
        for idx in range(current_bs):
            sample_id = sample_offset + idx
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
            result_map = {"baseline": baseline_result, "nsgrand": ns_result}
            if strong_symbolic is not None:
                strong_result = strong_symbolic.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
                result_map["strong_symbolic"] = strong_result
            point_trace_payloads: Dict[str, Dict[str, object]] = {}
            for decoder_name, result in result_map.items():
                record = _record_from_result(
                    sample_id=sample_id,
                    profile_name=profile_name,
                    profile_id=profile_id,
                    snr_db=snr_db,
                    decoder_name=decoder_name,
                    true_error_weight=true_error_weight,
                    truth_codeword=truth_codeword,
                    result=result,
                )
                records.append(record)
                trace_payload = {
                    "sample_id": int(sample_id),
                    "profile": profile_name,
                    "snr_db": snr_db,
                    "decoder": decoder_name,
                    "true_error_weight": true_error_weight,
                    "block_error": int(record["block_error"]),
                    "bit_errors": int(record["bit_errors"]),
                    "queries": int(record["queries"]),
                    "primary_queries": int(record["primary_queries"]),
                    "fallback_queries": int(record["fallback_queries"]),
                    "elapsed_ms": float(record["elapsed_ms"]),
                    "fallback_used": bool(record["fallback_used"]),
                    "confidence_prob": float(record["confidence_prob"]) if not np.isnan(record["confidence_prob"]) else None,
                    "predicted_overflow_prob": float(record["predicted_overflow_prob"]) if not np.isnan(record["predicted_overflow_prob"]) else None,
                    "gate_reason": record["gate_reason"],
                    "search_mode": record["search_mode"],
                    "trace": _flatten_trace(result.trace),
                }
                point_trace_payloads[decoder_name] = trace_payload
                if rng.random() < float(payload["trace_fraction"]):
                    trace_records.append(trace_payload)

                category = _trace_category_for_ns(record)
                limit = int(payload["interesting_traces_per_category"])
                if category is not None and interesting_counts.get(category, 0) < limit:
                    interesting_counts[category] = interesting_counts.get(category, 0) + 1
                    interesting_trace_records.append({**trace_payload, "category": category})
        processed += current_bs
        sample_offset += current_bs

    raw_df = pd.DataFrame(records)
    raw_df = raw_df.sort_values(["sample_id", "decoder"]).reset_index(drop=True)
    write_dataframe_csv(raw_df, raw_csv_path)
    if trace_records:
        append_jsonl(trace_records, traces_path)
    if interesting_trace_records:
        append_jsonl(interesting_trace_records, interesting_traces_path)

    paired_df = _build_paired_df(raw_df)
    if not paired_df.empty:
        write_dataframe_csv(paired_df, paired_csv_path)

    strong_pairwise_df = _build_pairwise_df(raw_df, ref_decoder="strong_symbolic", target_decoder="nsgrand")
    strong_pairwise_path = point_dir / "strong_vs_nsgrand_paired_records.csv"
    if not strong_pairwise_df.empty:
        write_dataframe_csv(strong_pairwise_df, strong_pairwise_path)

    summary_rows = []
    for _, sub_df in raw_df.groupby("decoder"):
        summary_rows.append(
            _summarize_decoder_frame(
                sub_df,
                profile_name=profile_name,
                profile_id=profile_id,
                snr_db=snr_db,
                code_n=code.n,
            )
        )

    point_artifacts: Dict[str, str] = {"raw_csv_path": str(raw_csv_path), "summary_path": str(summary_path)}
    if not paired_df.empty:
        point_artifacts["paired_csv_path"] = str(paired_csv_path)
    if not strong_pairwise_df.empty:
        point_artifacts["strong_vs_nsgrand_paired_csv_path"] = str(strong_pairwise_path)

    ns_df = raw_df[raw_df["decoder"] == "nsgrand"].copy()
    if not ns_df.empty:
        gate_summary = (
            ns_df.groupby(["gate_reason", "stage"], as_index=False)
            .agg(
                count=("sample_id", "count"),
                bler=("block_error", "mean"),
                avg_queries=("queries", "mean"),
                avg_elapsed_ms=("elapsed_ms", "mean"),
                fallback_rate=("fallback_used", "mean"),
            )
            .sort_values(["count", "gate_reason"], ascending=[False, True])
        )
        gate_path = point_dir / "nsgrand_gate_summary.csv"
        write_dataframe_csv(gate_summary, gate_path)
        point_artifacts["gate_summary_csv"] = str(gate_path)

        stage_weight_summary = (
            ns_df.groupby(["stage", "true_error_weight"], as_index=False)
            .agg(
                count=("sample_id", "count"),
                bler=("block_error", "mean"),
                avg_queries=("queries", "mean"),
                avg_elapsed_ms=("elapsed_ms", "mean"),
            )
            .sort_values(["stage", "true_error_weight"])
        )
        stage_weight_path = point_dir / "nsgrand_by_stage_weight.csv"
        write_dataframe_csv(stage_weight_summary, stage_weight_path)
        point_artifacts["stage_weight_csv"] = str(stage_weight_path)

        action_summary = (
            ns_df.groupby(["policy_action", "gate_reason", "stage", "search_mode"], as_index=False)
            .agg(
                count=("sample_id", "count"),
                bler=("block_error", "mean"),
                avg_queries=("queries", "mean"),
                avg_elapsed_ms=("elapsed_ms", "mean"),
                fallback_rate=("fallback_used", "mean"),
            )
            .sort_values(["count", "policy_action"], ascending=[False, True])
        )
        action_path = point_dir / "nsgrand_action_summary.csv"
        write_dataframe_csv(action_summary, action_path)
        point_artifacts["action_summary_csv"] = str(action_path)

    write_json({"rows": summary_rows, "artifacts": point_artifacts, "interesting_trace_counts": interesting_counts}, summary_path)
    return {
        "summary_path": str(summary_path),
        "raw_csv_path": str(raw_csv_path),
        "paired_csv_path": str(paired_csv_path) if not paired_df.empty else None,
        "strong_vs_nsgrand_paired_csv_path": str(strong_pairwise_path) if not strong_pairwise_df.empty else None,
        "rows": summary_rows,
    }


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
                    "interesting_traces_per_category": int(cfg.eval.interesting_traces_per_category),
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
                    "ai_weight_penalties": list(cfg.search.ai_weight_penalties),
                    "top_segments": int(cfg.search.top_segments),
                    "top_bits_extra": int(cfg.search.top_bits_extra),
                    "confidence_threshold": float(cfg.search.confidence_threshold),
                    "confidence_presearch_threshold": float(cfg.search.confidence_presearch_threshold),
                    "candidate_mass_threshold": float(cfg.search.candidate_mass_threshold),
                    "trace_top_attempts": int(cfg.search.trace_top_attempts),
                    "adaptive_pool_size": int(cfg.search.adaptive_pool_size),
                    "adaptive_max_weight": int(cfg.search.adaptive_max_weight),
                    "adaptive_budget": int(cfg.search.adaptive_budget),
                    "adaptive_top_segments": int(cfg.search.adaptive_top_segments),
                    "adaptive_top_bits_extra": int(cfg.search.adaptive_top_bits_extra),
                    "overflow_expand_threshold": float(cfg.search.overflow_expand_threshold),
                    "overflow_direct_fallback_threshold": float(cfg.search.overflow_direct_fallback_threshold),
                    "overflow_direct_action": str(cfg.search.overflow_direct_action),
                    "overflow_direct_confidence_ceiling": float(cfg.search.overflow_direct_confidence_ceiling),
                    "always_fallback_after_ai_fail": bool(cfg.search.always_fallback_after_ai_fail),
                    "ns_fallback_kind": str(cfg.search.ns_fallback_kind),
                    "fallback_pool_size": int(cfg.search.fallback_pool_size),
                    "fallback_max_weight": int(cfg.search.fallback_max_weight),
                    "fallback_budget": int(cfg.search.fallback_budget),
                    "fallback_weight_penalties": list(cfg.search.fallback_weight_penalties),
                    "evaluate_strong_symbolic": bool(cfg.search.evaluate_strong_symbolic),
                    "strong_pool_size": int(cfg.search.strong_pool_size),
                    "strong_max_weight": int(cfg.search.strong_max_weight),
                    "strong_budget": int(cfg.search.strong_budget),
                    "strong_weight_penalties": list(cfg.search.strong_weight_penalties),
                }
            )

    logger.info("Evaluating %d operating points", len(tasks))
    all_rows: List[Dict[str, object]] = []
    worker_count = max(1, int(cfg.resources.evaluation_workers))
    results = []
    if worker_count == 1:
        for payload in tasks:
            result = _evaluate_point_worker(payload)
            logger.info("Completed evaluation at %s", result["summary_path"])
            all_rows.extend(result["rows"])
            results.append(result)
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
                results.append(result)

    summary_df = pd.DataFrame(all_rows).sort_values(["profile", "snr_db", "decoder"]).reset_index(drop=True)
    write_dataframe_csv(summary_df, eval_root / "evaluation_summary.csv")
    write_json({"rows": all_rows}, eval_root / "evaluation_summary.json")

    raw_paths = sorted(eval_root.glob("profile_*/snr_*dB/raw_records.csv"))
    if raw_paths:
        raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
        if cfg.eval.export_global_raw_gzip:
            write_dataframe_csv(raw_df, eval_root / "all_raw_records.csv.gz")
        paired_df = _build_paired_df(raw_df)
        if not paired_df.empty and cfg.eval.export_global_raw_gzip:
            write_dataframe_csv(paired_df, eval_root / "all_paired_records.csv.gz")
        strong_pairwise_df = _build_pairwise_df(raw_df, ref_decoder="strong_symbolic", target_decoder="nsgrand")
        if not strong_pairwise_df.empty and cfg.eval.export_global_raw_gzip:
            write_dataframe_csv(strong_pairwise_df, eval_root / "strong_vs_nsgrand_paired_records.csv.gz")

        ns_df = raw_df[raw_df["decoder"] == "nsgrand"].copy()
        if not ns_df.empty:
            gate_summary = (
                ns_df.groupby(["profile", "snr_db", "gate_reason", "stage"], as_index=False)
                .agg(
                    count=("sample_id", "count"),
                    bler=("block_error", "mean"),
                    avg_queries=("queries", "mean"),
                    avg_elapsed_ms=("elapsed_ms", "mean"),
                    fallback_rate=("fallback_used", "mean"),
                )
                .sort_values(["profile", "snr_db", "count"], ascending=[True, True, False])
            )
            write_dataframe_csv(gate_summary, eval_root / "nsgrand_gate_summary.csv")

            action_summary = (
                ns_df.groupby(["profile", "snr_db", "policy_action", "gate_reason", "stage", "search_mode"], as_index=False)
                .agg(
                    count=("sample_id", "count"),
                    bler=("block_error", "mean"),
                    avg_queries=("queries", "mean"),
                    avg_elapsed_ms=("elapsed_ms", "mean"),
                    fallback_rate=("fallback_used", "mean"),
                )
                .sort_values(["profile", "snr_db", "count"], ascending=[True, True, False])
            )
            write_dataframe_csv(action_summary, eval_root / "nsgrand_action_summary.csv")

            bins = np.linspace(0.0, 1.0, int(cfg.analysis.calibration_bins) + 1)
            conf_df = ns_df[ns_df["confidence_prob"].notna()].copy()
            if not conf_df.empty:
                conf_df["confidence_bin"] = pd.cut(conf_df["confidence_prob"], bins=bins, include_lowest=True, duplicates="drop")
                conf_summary = (
                    conf_df.groupby(["profile", "snr_db", "confidence_bin"], observed=False)
                    .agg(
                        count=("sample_id", "count"),
                        success_rate=("decoder_success", "mean"),
                        bler=("block_error", "mean"),
                        fallback_rate=("fallback_used", "mean"),
                        avg_queries=("queries", "mean"),
                    )
                    .reset_index()
                )
                write_dataframe_csv(conf_summary, eval_root / "nsgrand_confidence_bins.csv")

            ov_df = ns_df[ns_df["predicted_overflow_prob"].notna()].copy()
            if not ov_df.empty:
                ov_df["overflow_bin"] = pd.cut(ov_df["predicted_overflow_prob"], bins=bins, include_lowest=True, duplicates="drop")
                ov_summary = (
                    ov_df.groupby(["profile", "snr_db", "overflow_bin"], observed=False)
                    .agg(
                        count=("sample_id", "count"),
                        bler=("block_error", "mean"),
                        fallback_rate=("fallback_used", "mean"),
                        avg_queries=("queries", "mean"),
                    )
                    .reset_index()
                )
                write_dataframe_csv(ov_summary, eval_root / "nsgrand_overflow_bins.csv")

            tail_ns = ns_df.sort_values(["elapsed_ms", "queries"], ascending=[False, False]).head(int(cfg.analysis.top_tail_cases))
            write_dataframe_csv(tail_ns, eval_root / "nsgrand_tail_cases_top.csv")

    manifest = {
        "num_points": len(tasks),
        "num_summary_rows": int(len(summary_df)),
        "points": sorted(str(path) for path in raw_paths) if raw_paths else [],
    }
    write_json(manifest, eval_root / "evaluation_manifest.json")
    return {"num_rows": int(len(summary_df)), "summary_csv": str(eval_root / "evaluation_summary.csv")}
