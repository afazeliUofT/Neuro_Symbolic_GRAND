from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config() -> Dict[str, Any]:
    return {
        "project": {"seed": 31415, "output_dir": "outputs/hybrid_bp_nsg_v14_pusch_cdl_full", "name": "hybrid_bp_nsg_v14"},
        "code": {
            "family": "sionna_nr_pusch_ldpc",
            "num_coded_bits": 512,
            "target_coderate": 0.5,
            "num_bits_per_symbol": 2,
            "num_layers": 1,
            "transport_k": 256,
            "strict_pcm_check": True,
            "seed": 31415,
            "use_sionna_pcm": True,
            "fallback_n_internal": 584,
            "fallback_m": 312,
        },
        "channel": {
            "modulation": "QPSK",
            "awgn": {},
            "cdl_c": {
                "carrier_frequency_hz": 3.5e9,
                "subcarrier_spacing_hz": 30000.0,
                "num_ofdm_symbols": 14,
                "fft_size": 72,
                "delay_spread_s": 1e-7,
                "speed_m_per_s": 0.0,
                "normalize_channel": True,
                "direction": "uplink",
                "model": "C",
                "perfect_csi": True,
            },
        },
        "data": {
            "train_samples": 36000,
            "val_samples": 7200,
            "shard_size": 250,
            "sample_failed_only": True,
            "bp_collect_iterations": 20,
            "failed_snr_db_grid": [1, 2, 3, 4, 5, 6],
            "failed_snr_probs": [0.10, 0.18, 0.25, 0.22, 0.16, 0.09],
            "profiles": ["AWGN", "CDL_C"],
            "profile_probs": [0.55, 0.45],
            "num_workers": 32,
            "parallel_chunk_size": 250,
            "filter_by_target_weight": True,
            "min_target_weight": 1,
            "max_target_weight": 32,
            "require_candidate_positive": True,
            "require_reachable": False,
            "max_attempt_multiplier": 80,
        },
        "model": {
            "graph_hidden_dim": 128,
            "graph_layers": 5,
            "transformer_heads": 4,
            "transformer_layers": 1,
            "top_k_tokens": 64,
            "num_segments": 8,
            "max_weight_class": 64,
            "rerank_list_size": 12,
        },
        "bp": {
            "hybrid_main_algorithm": "nms",
            "hybrid_main_iterations": 20,
            "strong_iterations": 50,
            "micro_iterations": 8,
            "nms_alpha": 0.8,
            "early_stop": True,
        },
        "rescue": {
            "mode": "ai",
            "target_basis": "bp",
            "fallback_target_basis": "channel_with_bp_punctures",
            "always_rescue_after_bp_fail": True,
            "try_bp_basis_fallback": False,
            "require_crc_for_accept": True,
            "fallback_accept_parity_without_crc": False,
            "gating_threshold": -1.0,
            "expanded_threshold": 0.2,
            "hopeless_threshold": -1.0,
            "rescue_threshold": 0.05,
            "pool_size": 64,
            "expanded_pool_size": 192,
            "top_k_bits": 128,
            "top_k_oscillation": 32,
            "top_k_unsat": 48,
            "max_standard_weight": 10,
            "max_expanded_weight": 32,
            "standard_budget": 1200,
            "expanded_budget": 3600,
            "direct_budget": 2200,
            "combo_pool_w_le3": 28,
            "combo_pool_w_gt3": 18,
            "enable_osd_repair": True,
            "osd_support_sizes": [64, 96, 128, 160, 192, 256, 320],
            "osd_jitter_passes": 2,
            "enable_greedy_repair": True,
            "greedy_repair_steps": 48,
            "greedy_repair_candidates": 8,
            "enable_micro_bp": True,
            "micro_candidate_cap": 8,
            "likely_weight_topk": 10,
            "rerank_extra_queries": 64,
            "rerank_use_net": True,
            "component_bonus": -0.35,
            "segment_bonus_scale": 0.08,
            "component_focus_scale": 0.1,
            "oscillation_focus_scale": 0.05,
            "rerank_prior_scale": 0.1,
            "micro_flip_scale": 1.1,
            "reachability_label_mode": "weight_only",
            "candidate_bank_pool_size": 160,
            "candidate_bank_top_k_bits": 128,
            "candidate_bank_top_k_unsat": 48,
            "candidate_bank_top_k_oscillation": 32,
            "candidate_bank_max_weight": 12,
            "candidate_bank_budget": 256,
            "candidate_bank_enable_greedy": True,
            "candidate_bank_greedy_steps": 24,
            "candidate_bank_greedy_candidates": 4,
            "candidate_bank_enable_osd": True,
            "candidate_bank_osd_support_sizes": [64, 96, 128, 160],
            "candidate_bank_osd_jitter_passes": 1,
            "oracle_candidate_max_weight": 32,
            "candidate_bank_inject_oracle_positive": True,
            "parallel_test_batch_size": 256,
        },
        "train": {
            "epochs": 48,
            "batch_size": 96,
            "lr": 8e-4,
            "weight_decay": 1e-5,
            "dropout": 0.08,
            "bit_pos_weight": 8.0,
            "reach_pos_weight": 8.0,
            "grad_clip": 1.0,
            "resume": False,
            "require_gpu": True,
            "mixed_precision": "bfloat16",
            "xla": False,
            "cpu_threads": 32,
            "preload_dataset": True,
            "loss_weights": {
                "bit": 1.0,
                "segment": 0.2,
                "weight": 0.2,
                "standard_reachable": 0.1,
                "expanded_reachable": 0.12,
                "rescueable": 0.1,
                "rerank": 0.75,
                "rank": 0.05,
            },
        },
        "eval": {
            "require_gpu": True,
            "mixed_precision": False,
            "xla": False,
            "cpu_threads": 32,
            "samples_per_point": 4000,
            "tail_samples_per_point": 30000,
            "target_frame_errors": 200,
            "max_samples_per_point": 200000,
            "stop_decoders": ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"],
            "snr_db_grid": [0, 1, 2, 3, 4, 5, 6],
            "profiles": ["AWGN", "CDL_C"],
        },
    }


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    cfg = deep_update(default_config(), data)
    cfg["_config_path"] = str(p)
    return cfg


def save_resolved_config(cfg: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir) / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
