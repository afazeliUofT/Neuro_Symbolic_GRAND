from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..decoders.bp import BPDecodeResult

PROFILE_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.max(np.abs(x))
    if denom < eps:
        return x
    return x / denom


def _build_unsat_history(code: LDPCCode, syndrome_trace: np.ndarray) -> np.ndarray:
    return syndrome_trace @ code.h


def _connected_suspect_components(code: LDPCCode, syndrome: np.ndarray) -> List[np.ndarray]:
    unsat_checks = set(np.flatnonzero(syndrome))
    if not unsat_checks:
        return []
    suspect_vars = set(np.flatnonzero(code.h[list(unsat_checks)].sum(axis=0)))
    components = []
    visited_v = set()
    for v0 in suspect_vars:
        if v0 in visited_v:
            continue
        frontier_v = {int(v0)}
        comp_v = set()
        comp_c = set()
        while frontier_v:
            next_c = set()
            for v in frontier_v:
                comp_v.add(v)
                visited_v.add(v)
                next_c.update(np.flatnonzero(code.h[:, v]))
            next_c &= unsat_checks
            comp_c |= next_c
            next_v = set()
            for c in next_c:
                next_v.update(np.flatnonzero(code.h[c]))
            frontier_v = {int(v) for v in next_v if v in suspect_vars and v not in comp_v}
        if comp_v:
            components.append(np.array(sorted(comp_v), dtype=np.int16))
    components.sort(key=lambda x: (-len(x), int(x[0])))
    return components


def make_rerank_candidates(code: LDPCCode, heuristic_order: np.ndarray, residual_mask: np.ndarray,
                           var_features: np.ndarray, max_candidates: int = 6, max_weight: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_idx = np.flatnonzero(residual_mask)
    candidates: List[np.ndarray] = []
    labels: List[int] = []
    seen = set()

    def add_mask(mask_idx: np.ndarray, label: int) -> None:
        idx_tuple = tuple(sorted(int(i) for i in np.asarray(mask_idx).tolist()))
        if len(idx_tuple) == 0 or idx_tuple in seen or len(candidates) >= max_candidates:
            return
        seen.add(idx_tuple)
        mask = np.zeros(code.n, dtype=np.uint8)
        mask[list(idx_tuple)] = 1
        candidates.append(mask)
        labels.append(label)

    if 0 < residual_idx.size <= max_weight:
        add_mask(residual_idx, 1)
    top = heuristic_order[: min(8, code.n)]
    for i in range(min(4, top.size)):
        add_mask(np.array([top[i]], dtype=np.int16), 0)
    if top.size >= 2:
        add_mask(np.array(top[:2], dtype=np.int16), 0)
        add_mask(np.array([top[0], top[2]], dtype=np.int16), 0)
    if top.size >= 3:
        add_mask(np.array(top[:3], dtype=np.int16), 0)
    if top.size >= 4:
        add_mask(np.array(top[:4], dtype=np.int16), 0)

    feat_list: List[np.ndarray] = []
    for cand in candidates:
        idx = np.flatnonzero(cand)
        vf = var_features[idx] if idx.size > 0 else np.zeros((1, var_features.shape[1]), dtype=np.float32)
        feat = np.array([
            idx.size,
            float(np.sum(vf[:, 0])),
            float(np.mean(vf[:, 0])),
            float(np.max(vf[:, 0])),
            float(np.sum(vf[:, 5])),
            float(np.mean(vf[:, 6])),
            float(np.sum(vf[:, 8])),
            float(np.mean(vf[:, 8])),
        ], dtype=np.float32)
        feat_list.append(feat)

    if not feat_list:
        return (
            np.zeros((max_candidates, 8), dtype=np.float32),
            np.zeros(max_candidates, dtype=np.uint8),
            np.zeros(max_candidates, dtype=np.uint8),
        )

    cand_feats = np.zeros((max_candidates, feat_list[0].size), dtype=np.float32)
    cand_labels = np.zeros(max_candidates, dtype=np.uint8)
    cand_valid = np.zeros(max_candidates, dtype=np.uint8)
    for i, (feat, lab) in enumerate(zip(feat_list, labels)):
        cand_feats[i] = feat
        cand_labels[i] = lab
        cand_valid[i] = 1
    return cand_feats, cand_labels, cand_valid


def build_rescue_features(code: LDPCCode, channel_llr: np.ndarray, bp_result: BPDecodeResult,
                          snr_db: float, profile: str, num_segments: int = 8) -> Dict[str, np.ndarray]:
    llr = np.asarray(channel_llr, dtype=np.float32)
    final_llr = np.asarray(bp_result.posterior_llr, dtype=np.float32)
    hard = np.asarray(bp_result.hard, dtype=np.uint8)
    syndrome = np.asarray(bp_result.syndrome, dtype=np.uint8)
    trace = bp_result.trace or {}
    llr_trace = np.asarray(trace.get("posterior_llr", final_llr[None, :]), dtype=np.float32)
    hard_trace = np.asarray(trace.get("hard", hard[None, :]), dtype=np.uint8)
    syndrome_trace = np.asarray(trace.get("syndrome", syndrome[None, :]), dtype=np.uint8)
    unsat_hist = _build_unsat_history(code, syndrome_trace).astype(np.float32)
    final_unsat = unsat_hist[-1]
    mean_unsat = np.mean(unsat_hist, axis=0)
    oscillation = np.sum(np.abs(np.diff(hard_trace.astype(np.int16), axis=0)), axis=0).astype(np.float32)
    llr_delta = final_llr - llr_trace[0]
    suspicion = (-np.abs(final_llr)
                 + 0.55 * final_unsat
                 + 0.35 * mean_unsat
                 + 0.20 * oscillation
                 + 0.15 * (hard != (llr < 0)).astype(np.float32))
    heuristic_order = np.argsort(-suspicion)
    inv_rank = np.empty(code.n, dtype=np.float32)
    inv_rank[heuristic_order] = np.linspace(1.0, 0.0, code.n, endpoint=False, dtype=np.float32)

    var_features = np.stack([
        _normalize(np.abs(final_llr)),
        np.sign(final_llr).astype(np.float32),
        hard.astype(np.float32),
        _normalize(np.abs(llr)),
        np.sign(llr).astype(np.float32),
        _normalize(final_unsat),
        _normalize(mean_unsat),
        _normalize(oscillation),
        inv_rank,
        _normalize(llr_delta),
        _normalize(code.deg_v.astype(np.float32)),
    ], axis=1).astype(np.float32)

    syndrome_mean = np.mean(syndrome_trace, axis=0).astype(np.float32)
    check_abs = np.array([np.mean(np.abs(final_llr[np.flatnonzero(code.h[c])])) for c in range(code.m)], dtype=np.float32)
    parity_tension = np.array([np.mean(hard[np.flatnonzero(code.h[c])]) for c in range(code.m)], dtype=np.float32)
    check_features = np.stack([
        syndrome.astype(np.float32),
        syndrome_mean,
        _normalize(check_abs),
        _normalize(parity_tension),
        _normalize(code.deg_c.astype(np.float32)),
    ], axis=1).astype(np.float32)

    components = _connected_suspect_components(code, syndrome)
    component_score = np.zeros(code.n, dtype=np.float32)
    for comp in components[:4]:
        component_score[comp] = np.maximum(component_score[comp], float(len(comp)))
    var_features = np.concatenate([var_features, _normalize(component_score)[:, None]], axis=1)

    seg_labels = np.zeros(num_segments, dtype=np.uint8)
    seg_size = int(np.ceil(code.n / num_segments))
    segment_index = np.zeros(code.n, dtype=np.int16)
    for s in range(num_segments):
        start = s * seg_size
        stop = min((s + 1) * seg_size, code.n)
        segment_index[heuristic_order[start:stop]] = s

    global_features = np.array([
        float(snr_db) / 10.0,
        float(PROFILE_TO_ID.get(profile, 0)) / max(1, len(PROFILE_TO_ID) - 1),
        float(syndrome.sum()) / max(1, code.m),
        float(bp_result.iterations_used) / max(1, hard_trace.shape[0]),
        float(np.mean(np.abs(final_llr) < 1.5)),
        float(np.mean(oscillation > 0)),
        float(np.mean(final_unsat > 0)),
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
    }


def build_training_labels(code: LDPCCode, feature_pack: Dict[str, np.ndarray], true_codeword: np.ndarray,
                          bp_result: BPDecodeResult, rescue_cfg: Dict[str, object], model_cfg: Dict[str, object]) -> Dict[str, np.ndarray]:
    residual_mask = (bp_result.hard ^ true_codeword).astype(np.uint8)
    residual_idx = np.flatnonzero(residual_mask)
    num_segments = int(model_cfg["num_segments"])
    seg_labels = np.zeros(num_segments, dtype=np.uint8)
    if residual_idx.size > 0:
        seg_labels[np.unique(feature_pack["segment_index"][residual_idx])] = 1
    max_weight_class = int(model_cfg["max_weight_class"])
    residual_weight = int(residual_idx.size)
    weight_label = min(residual_weight, max_weight_class + 1)
    order = feature_pack["heuristic_order"]
    top_std = set(int(x) for x in order[: int(rescue_cfg["pool_size"])] )
    top_exp = set(int(x) for x in order[: int(rescue_cfg["expanded_pool_size"])] )
    standard_reachable = int(0 < residual_weight <= int(rescue_cfg["max_standard_weight"]) and all(int(i) in top_std for i in residual_idx))
    expanded_reachable = int(0 < residual_weight <= int(rescue_cfg["max_expanded_weight"]) and all(int(i) in top_exp for i in residual_idx))
    rescueable = int((not standard_reachable) and expanded_reachable)
    candidate_feats, candidate_labels, candidate_valid = make_rerank_candidates(
        code=code,
        heuristic_order=order,
        residual_mask=residual_mask,
        var_features=feature_pack["var_features"],
        max_candidates=int(model_cfg.get("rerank_list_size", 6)),
        max_weight=int(rescue_cfg["max_expanded_weight"]),
    )
    return {
        "bit_labels": residual_mask.astype(np.uint8),
        "segment_labels": seg_labels,
        "weight_label": np.array(weight_label, dtype=np.int16),
        "standard_reachable": np.array(standard_reachable, dtype=np.uint8),
        "expanded_reachable": np.array(expanded_reachable, dtype=np.uint8),
        "rescueable": np.array(rescueable, dtype=np.uint8),
        "candidate_features": candidate_feats.astype(np.float32),
        "candidate_labels": candidate_labels.astype(np.uint8),
        "candidate_valid": candidate_valid.astype(np.uint8),
        "residual_weight": np.array(residual_weight, dtype=np.int16),
    }
