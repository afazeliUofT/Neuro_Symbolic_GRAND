from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..decoders.bp import BPDecodeResult

PROFILE_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = float(np.max(np.abs(x))) if x.size else 0.0
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / denom).astype(np.float32)


def _build_unsat_history(code: LDPCCode, syndrome_trace: np.ndarray) -> np.ndarray:
    return (syndrome_trace.astype(np.float32) @ code.h.astype(np.float32)).astype(np.float32)


def _connected_suspect_components(code: LDPCCode, syndrome: np.ndarray) -> List[np.ndarray]:
    unsat_checks = set(int(x) for x in np.flatnonzero(syndrome))
    if not unsat_checks:
        return []
    h = code.h
    suspect_vars = set(int(x) for x in np.flatnonzero(h[list(unsat_checks)].sum(axis=0)))
    components: List[np.ndarray] = []
    visited_v = set()
    for v0 in sorted(suspect_vars):
        if v0 in visited_v:
            continue
        frontier_v = {int(v0)}
        comp_v = set()
        while frontier_v:
            next_c = set()
            for v in frontier_v:
                comp_v.add(v)
                visited_v.add(v)
                next_c.update(int(c) for c in np.flatnonzero(h[:, v]))
            next_c &= unsat_checks
            next_v = set()
            for c in next_c:
                next_v.update(int(v) for v in np.flatnonzero(h[c]))
            frontier_v = {v for v in next_v if v in suspect_vars and v not in comp_v}
        if comp_v:
            components.append(np.array(sorted(comp_v), dtype=np.int16))
    components.sort(key=lambda x: (-len(x), int(x[0]) if len(x) else 0))
    return components


def make_grand_base(code: LDPCCode, channel_llr: np.ndarray, bp_result: BPDecodeResult, basis: str = "channel_with_bp_punctures") -> np.ndarray:
    """Return the hard vector to which GRAND noise masks are applied.

    GRAND should guess channel noise/correction, not the final failed BP residual. For punctured
    positions there is no channel observation, so the default uses BP posterior decisions there.
    """
    basis = str(basis or "channel_with_bp_punctures").lower()
    llr = np.asarray(channel_llr, dtype=np.float32).reshape(-1)
    channel_hard = (llr < 0).astype(np.uint8)
    if basis in {"bp", "bp_hard", "failed_bp"}:
        return np.asarray(bp_result.hard, dtype=np.uint8).copy()
    if basis in {"channel", "channel_hard"}:
        return channel_hard
    if basis in {"channel_with_bp_punctures", "channel_bp_punctures", "hybrid"}:
        base = channel_hard.copy()
        punc = code.punctured_positions
        if punc.size:
            base[punc] = np.asarray(bp_result.hard, dtype=np.uint8)[punc]
        return base
    if basis in {"posterior", "bp_transmitted_channel_punctured"}:
        base = np.asarray(bp_result.hard, dtype=np.uint8).copy()
        tx = code.tx_positions
        base[tx] = channel_hard[tx]
        return base
    raise ValueError(f"unknown GRAND target basis: {basis}")


def _mask_features(mask: np.ndarray, feature_pack: Dict[str, np.ndarray]) -> np.ndarray:
    idx = np.flatnonzero(mask)
    vf = feature_pack["var_features"][idx] if idx.size > 0 else np.zeros((1, feature_pack["var_features"].shape[1]), dtype=np.float32)
    return np.array([
        idx.size,
        float(np.sum(vf[:, 0])),
        float(np.mean(vf[:, 0])),
        float(np.max(vf[:, 0])),
        float(np.sum(vf[:, 5])),
        float(np.mean(vf[:, 6])),
        float(np.sum(vf[:, 8])),
        float(np.mean(vf[:, 8])),
    ], dtype=np.float32)


def make_rerank_candidates(
    code: LDPCCode,
    heuristic_order: np.ndarray,
    target_mask: np.ndarray,
    var_features: np.ndarray,
    max_candidates: int = 8,
    max_weight: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_idx = np.flatnonzero(target_mask)
    candidates: List[np.ndarray] = []
    labels: List[int] = []
    seen = set()

    def add_mask(mask_idx: np.ndarray, label: int) -> None:
        idx_tuple = tuple(sorted(int(i) for i in np.asarray(mask_idx).reshape(-1).tolist()))
        if not idx_tuple or idx_tuple in seen or len(candidates) >= max_candidates:
            return
        seen.add(idx_tuple)
        mask = np.zeros(code.n, dtype=np.uint8)
        mask[list(idx_tuple)] = 1
        candidates.append(mask)
        labels.append(int(label))

    if 0 < target_idx.size <= max_weight:
        add_mask(target_idx, 1)

    top = np.asarray(heuristic_order[: min(max(12, max_candidates + 4), code.n)], dtype=np.int16)
    for i in range(min(6, top.size)):
        add_mask(np.array([top[i]], dtype=np.int16), 0)
    if top.size >= 2:
        add_mask(np.array(top[:2], dtype=np.int16), 0)
        add_mask(np.array([top[0], top[2 if top.size > 2 else 1]], dtype=np.int16), 0)
    if top.size >= 3:
        add_mask(np.array(top[:3], dtype=np.int16), 0)
    if top.size >= 4:
        add_mask(np.array(top[:4], dtype=np.int16), 0)

    feat_list = [_mask_features(cand, {"var_features": var_features}) for cand in candidates]
    cand_feats = np.zeros((max_candidates, 8), dtype=np.float32)
    cand_labels = np.zeros(max_candidates, dtype=np.uint8)
    cand_valid = np.zeros(max_candidates, dtype=np.uint8)
    for i, feat in enumerate(feat_list[:max_candidates]):
        cand_feats[i] = feat
        cand_labels[i] = labels[i]
        cand_valid[i] = 1
    return cand_feats, cand_labels, cand_valid


def build_rescue_features(
    code: LDPCCode,
    channel_llr: np.ndarray,
    bp_result: BPDecodeResult,
    snr_db: float,
    profile: str,
    num_segments: int = 8,
    target_basis: str = "channel_with_bp_punctures",
) -> Dict[str, np.ndarray]:
    llr = np.asarray(channel_llr, dtype=np.float32).reshape(-1)
    final_llr = np.asarray(bp_result.posterior_llr, dtype=np.float32).reshape(-1)
    hard = np.asarray(bp_result.hard, dtype=np.uint8).reshape(-1)
    channel_hard = (llr < 0).astype(np.uint8)
    base_hard = make_grand_base(code, llr, bp_result, target_basis)
    base_syndrome = code.syndrome(base_hard)
    syndrome = np.asarray(bp_result.syndrome, dtype=np.uint8)
    trace = bp_result.trace or {}
    llr_trace = np.asarray(trace.get("posterior_llr", final_llr[None, :]), dtype=np.float32)
    hard_trace = np.asarray(trace.get("hard", hard[None, :]), dtype=np.uint8)
    syndrome_trace = np.asarray(trace.get("syndrome", syndrome[None, :]), dtype=np.uint8)
    if llr_trace.ndim == 1:
        llr_trace = llr_trace[None, :]
    if hard_trace.ndim == 1:
        hard_trace = hard_trace[None, :]
    if syndrome_trace.ndim == 1:
        syndrome_trace = syndrome_trace[None, :]

    unsat_hist = _build_unsat_history(code, syndrome_trace)
    final_unsat = unsat_hist[-1] if len(unsat_hist) else np.zeros(code.n, dtype=np.float32)
    mean_unsat = np.mean(unsat_hist, axis=0) if len(unsat_hist) else final_unsat
    base_unsat = (base_syndrome.astype(np.float32) @ code.h.astype(np.float32)).astype(np.float32)
    oscillation = np.sum(np.abs(np.diff(hard_trace.astype(np.int16), axis=0)), axis=0).astype(np.float32) if hard_trace.shape[0] > 1 else np.zeros(code.n, dtype=np.float32)
    llr_delta = final_llr - llr_trace[0]
    disagreement = (hard != channel_hard).astype(np.float32)
    base_disagreement = (base_hard != hard).astype(np.float32)
    punctured = np.zeros(code.n, dtype=np.float32)
    if code.punctured_positions.size:
        punctured[code.punctured_positions] = 1.0

    components = _connected_suspect_components(code, syndrome | base_syndrome)
    component_score = np.zeros(code.n, dtype=np.float32)
    for comp in components[:6]:
        component_score[comp] = np.maximum(component_score[comp], float(len(comp)))

    # Channel-aligned suspicion: prioritize low channel reliability, base syndrome participation,
    # BP unsatisfied-check participation, oscillation, and channel/BP disagreement.
    abs_ch = np.abs(llr)
    abs_final = np.abs(final_llr)
    unreliability = 1.0 - np.clip(_normalize(abs_ch), 0.0, 1.0)
    suspicion = (
        1.40 * unreliability
        + 0.75 * _normalize(base_unsat)
        + 0.60 * _normalize(final_unsat)
        + 0.35 * _normalize(mean_unsat)
        + 0.30 * _normalize(oscillation)
        + 0.25 * disagreement
        + 0.20 * base_disagreement
        + 0.20 * _normalize(component_score)
        + 0.15 * punctured * (1.0 - np.clip(_normalize(abs_final), 0.0, 1.0))
    ).astype(np.float32)

    heuristic_order = np.argsort(-suspicion).astype(np.int16)
    inv_rank = np.empty(code.n, dtype=np.float32)
    inv_rank[heuristic_order] = np.linspace(1.0, 0.0, code.n, endpoint=False, dtype=np.float32)

    var_features = np.stack([
        _normalize(abs_final),
        np.sign(final_llr).astype(np.float32),
        hard.astype(np.float32),
        _normalize(abs_ch),
        np.sign(llr).astype(np.float32),
        _normalize(final_unsat),
        _normalize(mean_unsat),
        _normalize(oscillation),
        inv_rank,
        _normalize(llr_delta),
        _normalize(code.deg_v.astype(np.float32)),
        _normalize(base_unsat),
        base_hard.astype(np.float32),
        disagreement.astype(np.float32),
        punctured.astype(np.float32),
        _normalize(component_score),
    ], axis=1).astype(np.float32)

    syndrome_mean = np.mean(syndrome_trace, axis=0).astype(np.float32) if len(syndrome_trace) else syndrome.astype(np.float32)
    check_abs = np.zeros(code.m, dtype=np.float32)
    parity_tension = np.zeros(code.m, dtype=np.float32)
    for c in range(code.m):
        idx = np.flatnonzero(code.h[c])
        if idx.size:
            check_abs[c] = float(np.mean(np.abs(final_llr[idx])))
            parity_tension[c] = float(np.mean(hard[idx]))
    check_features = np.stack([
        syndrome.astype(np.float32),
        base_syndrome.astype(np.float32),
        syndrome_mean,
        _normalize(check_abs),
        _normalize(parity_tension),
        _normalize(code.deg_c.astype(np.float32)),
    ], axis=1).astype(np.float32)

    seg_size = int(np.ceil(code.n / max(1, num_segments)))
    segment_index = np.zeros(code.n, dtype=np.int16)
    for s in range(num_segments):
        start = s * seg_size
        stop = min((s + 1) * seg_size, code.n)
        segment_index[heuristic_order[start:stop]] = s

    global_features = np.array([
        float(snr_db) / 10.0,
        float(PROFILE_TO_ID.get(profile, 0)) / max(1, len(PROFILE_TO_ID) - 1),
        float(syndrome.sum()) / max(1, code.m),
        float(base_syndrome.sum()) / max(1, code.m),
        float(bp_result.iterations_used) / max(1, hard_trace.shape[0]),
        float(np.mean(np.abs(final_llr) < 1.5)),
        float(np.mean(oscillation > 0)),
        float(np.mean(base_unsat > 0)),
        code.rate,
    ], dtype=np.float32)

    return {
        "var_features": var_features,
        "check_features": check_features,
        "global_features": global_features,
        "heuristic_order": heuristic_order.astype(np.int16),
        "segment_index": segment_index.astype(np.int16),
        "components": components,
        "suspicion": suspicion.astype(np.float32),
        "grand_base_hard": base_hard.astype(np.uint8),
        "grand_base_syndrome": base_syndrome.astype(np.uint8),
        "channel_hard": channel_hard.astype(np.uint8),
    }


def build_training_labels(
    code: LDPCCode,
    feature_pack: Dict[str, np.ndarray],
    true_codeword: np.ndarray,
    bp_result: BPDecodeResult,
    rescue_cfg: Dict[str, object],
    model_cfg: Dict[str, object],
) -> Dict[str, np.ndarray]:
    true_codeword = np.asarray(true_codeword, dtype=np.uint8).reshape(-1)
    base_hard = np.asarray(feature_pack["grand_base_hard"], dtype=np.uint8)
    # The v11 target is the channel-aligned GRAND-base correction.
    target_mask = (base_hard ^ true_codeword).astype(np.uint8)
    target_idx = np.flatnonzero(target_mask)
    bp_residual_mask = (np.asarray(bp_result.hard, dtype=np.uint8) ^ true_codeword).astype(np.uint8)

    num_segments = int(model_cfg["num_segments"])
    seg_labels = np.zeros(num_segments, dtype=np.uint8)
    if target_idx.size > 0:
        seg_labels[np.unique(feature_pack["segment_index"][target_idx])] = 1

    max_weight_class = int(model_cfg["max_weight_class"])
    target_weight = int(target_idx.size)
    weight_label = min(target_weight, max_weight_class + 1)
    order = feature_pack["heuristic_order"]
    top_std = set(int(x) for x in order[: int(rescue_cfg["pool_size"])])
    top_exp = set(int(x) for x in order[: int(rescue_cfg["expanded_pool_size"])])
    standard_reachable = int(0 < target_weight <= int(rescue_cfg["max_standard_weight"]) and all(int(i) in top_std for i in target_idx))
    expanded_reachable = int(0 < target_weight <= int(rescue_cfg["max_expanded_weight"]) and all(int(i) in top_exp for i in target_idx))
    rescueable = int((not standard_reachable) and expanded_reachable)

    candidate_feats, candidate_labels, candidate_valid = make_rerank_candidates(
        code=code,
        heuristic_order=order,
        target_mask=target_mask,
        var_features=feature_pack["var_features"],
        max_candidates=int(model_cfg.get("rerank_list_size", 8)),
        max_weight=int(rescue_cfg["max_expanded_weight"]),
    )
    return {
        "bit_labels": target_mask.astype(np.uint8),
        "segment_labels": seg_labels.astype(np.uint8),
        "weight_label": np.array(weight_label, dtype=np.int16),
        "standard_reachable": np.array(standard_reachable, dtype=np.uint8),
        "expanded_reachable": np.array(expanded_reachable, dtype=np.uint8),
        "rescueable": np.array(rescueable, dtype=np.uint8),
        "candidate_features": candidate_feats.astype(np.float32),
        "candidate_labels": candidate_labels.astype(np.uint8),
        "candidate_valid": candidate_valid.astype(np.uint8),
        "target_weight": np.array(target_weight, dtype=np.int16),
        "bp_residual_weight": np.array(int(bp_residual_mask.sum()), dtype=np.int16),
        "channel_basis": np.array(1, dtype=np.uint8),
    }
