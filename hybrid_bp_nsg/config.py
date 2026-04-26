from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "project": {"seed": 1234, "output_dir": "outputs/hybrid_bp_nsg_v13_pusch_cdl_default"},
    "code": {"family": "peg_ldpc", "k": 32, "n": 64, "seed": 1234},
    "data": {
        "train_samples": 1000,
        "val_samples": 200,
        "shard_size": 250,
        "sample_failed_only": True,
        "bp_collect_iterations": 20,
        "failed_snr_db_grid": [0, 1, 2, 3],
        "failed_snr_probs": [0.4, 0.3, 0.2, 0.1],
        "profiles": ["AWGN", "CDL_C"],
    },
    "channel": {
        "modulation": "QPSK",
        "awgn": {},
        "cdl_c": {
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 30e3,
            "num_ofdm_symbols": 14,
            "fft_size": 72,
            "cyclic_prefix_length": 0,
            "delay_spread_s": 100e-9,
            "speed_m_per_s": 0.0,
            "normalize_channel": True,
            "direction": "uplink",
            "model": "C",
            "perfect_csi": True,
        },
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
        "parallel_test_batch_size": 256,
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
        "candidate_bank_inject_oracle_positive": False,
    },
    "eval": {
        "samples_per_point": 4000,
        "tail_samples_per_point": 30000,
        "target_frame_errors": 200,
        "max_samples_per_point": 200000,
        "stop_decoders": ["hybrid_bp_nsg", "bp_nms_20", "bp_nms_50"],
        "snr_db_grid": [0, 1, 2, 3],
        "profiles": ["AWGN", "CDL_C"],
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


def normalize_snr_sampling(data: Dict[str, Any]) -> None:
    """Make failed-SNR sampling probabilities consistent with the configured grid.

    Earlier package versions accidentally shipped smoke/tail configs where `failed_snr_db_grid`
    was shortened but `failed_snr_probs` still had the 11-entry full-run vector.
    NumPy then raises `ValueError: a and p must have same size` during
    `rng.choice(grid, p=probs)`.  This normalizer both fixes old configs and
    makes future shortened smoke/tail grids robust.

    If the probability list is longer than the grid and the grid entries look
    like integer SNR points, we interpret the probability vector as the full
    dense 0,1,2,... prior and select the probabilities corresponding to the
    requested grid.  Otherwise we crop/pad conservatively and normalize.
    """
    grid = list(data.get("failed_snr_db_grid", []) or [])
    if not grid:
        data["failed_snr_db_grid"] = [0.0]
        data["failed_snr_probs"] = [1.0]
        return

    raw_probs = data.get("failed_snr_probs", None)
    if raw_probs is None:
        probs = [1.0 for _ in grid]
        note = "uniform_default"
    else:
        probs = [float(x) for x in list(raw_probs)]
        note = "as_configured"

    if len(probs) != len(grid):
        selected = None
        # Common FIR case: full-run prior has entries for integer SNR 0..K,
        # while smoke/tail uses a subset such as [0, 2, 4] or [4..10].
        try:
            idx = [int(round(float(x))) for x in grid]
            if all(abs(float(g) - i) < 1e-6 and 0 <= i < len(probs) for g, i in zip(grid, idx)):
                selected = [probs[i] for i in idx]
                note = f"selected_from_dense_prior_len_{len(probs)}"
        except Exception:
            selected = None
        if selected is None:
            if len(probs) > len(grid):
                selected = probs[:len(grid)]
                note = f"cropped_from_len_{len(probs)}"
            else:
                selected = probs + [1.0 for _ in range(len(grid) - len(probs))]
                note = f"padded_from_len_{len(probs)}"
        probs = selected

    # Ensure all entries are finite and non-negative, then normalize.
    clean = []
    for x in probs:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if not (v >= 0.0) or v == float("inf") or v == float("-inf"):
            v = 0.0
        clean.append(v)
    total = sum(clean)
    if total <= 0.0:
        clean = [1.0 / len(grid) for _ in grid]
        note = "uniform_after_invalid_probs"
    else:
        clean = [float(x) / total for x in clean]

    data["failed_snr_db_grid"] = [float(x) if isinstance(x, float) else x for x in grid]
    data["failed_snr_probs"] = clean
    data["failed_snr_probs_note"] = note


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalize_snr_sampling(cfg.setdefault("data", {}))
    code = cfg.setdefault("code", {})
    fam = str(code.get("family", "")).lower()
    if fam in {"sionna_nr_pusch_ldpc", "sionna_nr_pusch", "nr_pusch_ldpc"}:
        if "num_coded_bits" in code:
            code["n"] = int(code["num_coded_bits"])
        if code.get("target_tb_size", None) is None:
            code.pop("k", None)
        else:
            code["k"] = int(code["target_tb_size"])
        code.pop("align_to_pcm_length", None)
    return cfg


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(DEFAULTS, user_cfg)
    cfg["_config_path"] = str(path)
    return validate_config(cfg)
