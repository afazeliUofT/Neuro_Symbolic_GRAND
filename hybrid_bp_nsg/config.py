from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "project": {"seed": 1234, "output_dir": "outputs/hybrid_bp_nsg_v11"},
    "code": {"family": "peg_ldpc", "k": 32, "n": 64, "seed": 1234},
    "data": {
        "train_samples": 1000,
        "val_samples": 200,
        "shard_size": 250,
        "sample_failed_only": True,
        "bp_collect_iterations": 20,
        "failed_snr_db_grid": [0, 1, 2, 3],
        "failed_snr_probs": [0.4, 0.3, 0.2, 0.1],
        "profiles": ["A"],
    },
    "train": {
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.05,
        "bit_pos_weight": 4.0,
        "reach_pos_weight": 4.0,
        "grad_clip": 1.0,
        "resume": True,
        "require_gpu": False,
        "mixed_precision": False,
        "preload_dataset": True,
        "loss_weights": {
            "bit": 1.0,
            "segment": 0.2,
            "weight": 0.2,
            "standard_reachable": 0.2,
            "expanded_reachable": 0.2,
            "rescueable": 0.2,
            "rerank": 0.2,
            "rank": 0.1,
        },
    },
    "model": {
        "graph_hidden_dim": 64,
        "graph_layers": 3,
        "transformer_heads": 4,
        "transformer_layers": 1,
        "top_k_tokens": 32,
        "num_segments": 8,
        "max_weight_class": 32,
        "rerank_list_size": 8,
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
        "target_basis": "channel_with_bp_punctures",
        "always_rescue_after_bp_fail": True,
        "try_bp_basis_fallback": True,
        "gating_threshold": -1.0,
        "expanded_threshold": 0.2,
        "hopeless_threshold": -1.0,
        "rescue_threshold": 0.05,
        "micro_trigger_threshold": 0.2,
        "pool_size": 48,
        "expanded_pool_size": 160,
        "top_k_bits": 96,
        "top_k_oscillation": 32,
        "top_k_unsat": 48,
        "max_standard_weight": 10,
        "max_expanded_weight": 32,
        "standard_budget": 800,
        "expanded_budget": 2400,
        "direct_budget": 1600,
        "combo_pool_w_le3": 24,
        "combo_pool_w_gt3": 16,
        "enable_osd_repair": True,
        "osd_support_sizes": [96, 128, 192, 256, 320],
        "osd_jitter_passes": 1,
        "enable_greedy_repair": True,
        "greedy_repair_steps": 32,
        "greedy_repair_candidates": 6,
        "enable_micro_bp": True,
        "micro_candidate_cap": 6,
        "likely_weight_topk": 8,
        "rerank_extra_queries": 32,
        "rerank_use_net": True,
        "weight_penalties": [0.0, 0.0, 0.08, 0.20, 0.38, 0.60, 0.85, 1.12, 1.45, 1.80],
    },
    "eval": {
        "samples_per_point": 1000,
        "tail_samples_per_point": 5000,
        "target_frame_errors": 100,
        "max_samples_per_point": 10000,
        "stop_decoders": ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"],
        "snr_db_grid": [0, 1, 2, 3],
        "profiles": ["A"],
        "require_gpu": False,
        "mixed_precision": False,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(DEFAULTS, user_cfg)
    cfg["_config_path"] = str(path)
    return cfg
