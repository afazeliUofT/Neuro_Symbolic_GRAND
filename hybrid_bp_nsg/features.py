from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .bp import BPDecodeResult
from .code import LDPCCode


def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 1.0
    return ((x - mu) / max(sd, eps)).astype(np.float32)


def make_grand_base(code: LDPCCode, llr: np.ndarray, bp_result: BPDecodeResult, target_basis: str) -> np.ndarray:
    target_basis = str(target_basis).lower()
    if target_basis in {"bp", "bp_hard", "failed_bp", "last_bp"}:
        return np.asarray(bp_result.hard, dtype=np.uint8).copy()
    channel_hard = (np.asarray(llr, dtype=np.float32) < 0).astype(np.uint8)
    if target_basis in {"channel", "channel_hard"}:
        return channel_hard
    if target_basis in {"channel_with_bp_punctures", "channel_bp_punctures"}:
        out = channel_hard.copy()
        if code.punctured_positions.size:
            out[code.punctured_positions] = np.asarray(bp_result.hard, dtype=np.uint8)[code.punctured_positions]
        return out
    return np.asarray(bp_result.hard, dtype=np.uint8).copy()


def _unsat_per_variable(code: LDPCCode, syndrome: np.ndarray) -> np.ndarray:
    return (code.h.T.astype(np.float32) @ np.asarray(syndrome, dtype=np.float32)).astype(np.float32)


def _components(code: LDPCCode, variable_score: np.ndarray, top: int = 96) -> np.ndarray:
    # Lightweight connected-component suspicion: diffuse unsatisfied-check scores once through Tanner graph.
    v = np.asarray(variable_score, dtype=np.float32)
    if v.size == 0:
        return v
    c = code.h.astype(np.float32) @ v
    c = c / np.maximum(code.deg_c, 1.0)
    out = code.h.T.astype(np.float32) @ c
    out = out / np.maximum(code.deg_v, 1.0)
    return out.astype(np.float32)


def build_rescue_features(code: LDPCCode, channel_llr: np.ndarray, bp_result: BPDecodeResult, snr_db: float, profile: str,
                          num_segments: int = 8, target_basis: str = "bp") -> Dict[str, np.ndarray]:
    llr = np.asarray(channel_llr, dtype=np.float32).reshape(-1)
    final_llr = np.asarray(bp_result.posterior_llr, dtype=np.float32).reshape(-1)
    hard = np.asarray(bp_result.hard, dtype=np.uint8).reshape(-1)
    channel_hard = (llr < 0).astype(np.uint8)
    base_hard = make_grand_base(code, llr, bp_result, target_basis)
    base_syndrome = code.syndrome(base_hard)
    syndrome = np.asarray(bp_result.syndrome, dtype=np.uint8)
    trace = bp_result.trace or {}
    hard_trace = np.asarray(trace.get("hard", hard[None, :]), dtype=np.uint8)
    llr_trace = np.asarray(trace.get("posterior_llr", final_llr[None, :]), dtype=np.float32)
    syn_trace = np.asarray(trace.get("syndrome", syndrome[None, :]), dtype=np.uint8)
    unsat = _unsat_per_variable(code, syndrome)
    base_unsat = _unsat_per_variable(code, base_syndrome)
    unsat_history = np.zeros(code.n, dtype=np.float32)
    for s in syn_trace:
        unsat_history += _unsat_per_variable(code, s)
    if syn_trace.shape[0]:
        unsat_history /= float(syn_trace.shape[0])
    oscillation = np.sum(np.abs(np.diff(hard_trace.astype(np.int16), axis=0)), axis=0).astype(np.float32) if hard_trace.shape[0] > 1 else np.zeros(code.n, dtype=np.float32)
    llr_delta = final_llr - llr_trace[0]
    disagreement = (hard != channel_hard).astype(np.float32)
    base_disagreement = (base_hard != hard).astype(np.float32)
    punctured = np.zeros(code.n, dtype=np.float32)
    if code.punctured_positions.size:
        punctured[code.punctured_positions] = 1.0
    suspicion = (
        1.25 * _normalize(base_unsat)
        + 0.80 * _normalize(unsat_history)
        + 0.65 * _normalize(oscillation)
        - 0.55 * _normalize(np.abs(final_llr))
        + 0.35 * disagreement
        + 0.20 * base_disagreement
        + 0.10 * punctured
    ).astype(np.float32)
    component_score = _components(code, np.maximum(suspicion, 0.0))
    # Lower cost is earlier GRAND order.
    bit_cost = (
        0.72 * _normalize(np.abs(final_llr))
        + 0.28 * _normalize(np.abs(llr))
        - 0.85 * _normalize(base_unsat)
        - 0.35 * _normalize(oscillation)
        - 0.25 * _normalize(component_score)
        + 0.10 * punctured
    ).astype(np.float32)
    heuristic_order = np.argsort(bit_cost).astype(np.int16)
    seg_size = int(math.ceil(code.n / max(1, num_segments)))
    segment_index = np.minimum(np.arange(code.n) // seg_size, num_segments - 1).astype(np.int16)
    profile_id = 0 if str(profile).upper().replace("-", "_") == "AWGN" else 1
    var_features = np.stack([
        _normalize(llr),
        _normalize(final_llr),
        _normalize(np.abs(llr)),
        _normalize(np.abs(final_llr)),
        hard.astype(np.float32),
        base_hard.astype(np.float32),
        channel_hard.astype(np.float32),
        _normalize(unsat),
        _normalize(base_unsat),
        _normalize(unsat_history),
        _normalize(oscillation),
        _normalize(llr_delta),
        _normalize(code.deg_v.astype(np.float32)),
        disagreement.astype(np.float32),
        punctured.astype(np.float32),
        _normalize(component_score),
    ], axis=-1).astype(np.float32)
    # Check features.
    abs_final = np.abs(final_llr)
    check_mean_abs = np.zeros(code.m, dtype=np.float32)
    check_min_abs = np.zeros(code.m, dtype=np.float32)
    for i, vs in enumerate(code.cn_neighbors):
        if len(vs):
            vals = abs_final[vs]
            check_mean_abs[i] = float(vals.mean())
            check_min_abs[i] = float(vals.min())
    check_features = np.stack([
        syndrome.astype(np.float32),
        base_syndrome.astype(np.float32),
        _normalize(code.deg_c.astype(np.float32)),
        _normalize(check_mean_abs),
        _normalize(check_min_abs),
        np.full(code.m, float(profile_id), dtype=np.float32),
    ], axis=-1).astype(np.float32)
    global_features = np.array([
        float(snr_db) / 10.0,
        float(profile_id),
        float(syndrome.sum()) / max(1, code.m),
        float(base_syndrome.sum()) / max(1, code.m),
        float(np.mean(np.abs(llr))),
        float(np.mean(np.abs(final_llr))),
        float(np.mean(oscillation)),
        float(code.rate),
        float(len(code.punctured_positions)) / max(1, code.n),
    ], dtype=np.float32)
    return {
        "var_features": var_features,
        "check_features": check_features,
        "global_features": global_features,
        "heuristic_order": heuristic_order,
        "bit_cost": bit_cost.astype(np.float32),
        "segment_index": segment_index,
        "suspicion": suspicion.astype(np.float32),
        "component_score": component_score.astype(np.float32),
        "grand_base_hard": base_hard.astype(np.uint8),
        "grand_base_syndrome": base_syndrome.astype(np.uint8),
        "channel_hard": channel_hard.astype(np.uint8),
        "profile_id": np.array(profile_id, dtype=np.int16),
        "snr_db": np.array(float(snr_db), dtype=np.float32),
    }


def candidate_feature_vector(code: LDPCCode, llr_internal: np.ndarray, base_hard: np.ndarray, codeword: np.ndarray,
                             mask: np.ndarray, feature_pack: Dict[str, np.ndarray], prior_score: float, crc_ok: bool) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
    idx = np.flatnonzero(mask)
    if idx.size:
        abs_llr = np.abs(llr_internal[idx])
        susp = feature_pack["suspicion"][idx]
        comp = feature_pack["component_score"][idx]
        cost = feature_pack["bit_cost"][idx]
    else:
        abs_llr = np.array([0.0], dtype=np.float32)
        susp = np.array([0.0], dtype=np.float32)
        comp = np.array([0.0], dtype=np.float32)
        cost = np.array([0.0], dtype=np.float32)
    syn = code.syndrome(codeword)
    return np.array([
        float(idx.size) / max(1, code.n),
        float(idx.size),
        float(np.sum(cost)),
        float(np.mean(cost)),
        float(np.min(abs_llr)),
        float(np.mean(abs_llr)),
        float(np.max(abs_llr)),
        float(np.mean(susp)),
        float(np.max(susp)),
        float(np.mean(comp)),
        float(syn.sum()) / max(1, code.m),
        float(prior_score),
        float(crc_ok),
        float(code.crc_check_internal(codeword)),
    ], dtype=np.float32)


def candidate_pool_from_feature_pack(feature_pack: Dict[str, np.ndarray], max_weight: int, budget: int,
                                     pool_size: int, combo_pool_w_le3: int = 28,
                                     combo_pool_w_gt3: int = 18) -> List[Tuple[np.ndarray, float, str]]:
    order = [int(x) for x in feature_pack["heuristic_order"][: int(pool_size)].tolist()]
    bit_cost = np.asarray(feature_pack["bit_cost"], dtype=np.float32)
    out: List[Tuple[np.ndarray, float, str]] = []
    seen: set[Tuple[int, ...]] = set()
    def add(mask_idx: Sequence[int], source: str) -> None:
        if len(out) >= budget:
            return
        key = tuple(sorted(set(int(i) for i in mask_idx)))
        if not key or key in seen:
            return
        seen.add(key)
        score = float(np.sum(bit_cost[list(key)])) + 0.2 * len(key)
        out.append((np.array(key, dtype=np.int32), score, source))
    # Low weights, ordered by bit cost.
    for w in range(1, min(max_weight, 4) + 1):
        psize = combo_pool_w_le3 if w <= 3 else combo_pool_w_gt3
        for comb in itertools.combinations(order[: min(psize, len(order))], w):
            add(comb, f"combo_w{w}")
            if len(out) >= budget:
                return out
    return out


def gf2_solve(Hs: np.ndarray, syndrome: np.ndarray) -> np.ndarray | None:
    A = np.asarray(Hs, dtype=np.uint8).copy() & 1
    b = np.asarray(syndrome, dtype=np.uint8).reshape(-1, 1).copy() & 1
    m, n = A.shape
    Ab = np.concatenate([A, b], axis=1)
    pivots: List[int] = []
    row = 0
    for col in range(n):
        piv = None
        candidates = np.flatnonzero(Ab[row:, col])
        if candidates.size:
            piv = int(row + candidates[0])
        if piv is None:
            continue
        if piv != row:
            Ab[[row, piv]] = Ab[[piv, row]]
        for r in range(m):
            if r != row and Ab[r, col]:
                Ab[r, :] ^= Ab[row, :]
        pivots.append(col)
        row += 1
        if row >= m:
            break
    # Inconsistent rows.
    for r in range(row, m):
        if not Ab[r, :n].any() and Ab[r, n]:
            return None
    x = np.zeros(n, dtype=np.uint8)
    for r, col in enumerate(pivots):
        x[col] = Ab[r, n]
    return x


def syndrome_osd_candidates(code: LDPCCode, base_hard: np.ndarray, bit_cost: np.ndarray,
                            support_sizes: Sequence[int], jitter_passes: int = 1,
                            rng: np.random.Generator | None = None) -> List[Tuple[np.ndarray, float, str]]:
    syn = code.syndrome(base_hard)
    if syn.sum() == 0:
        return []
    rng = rng or np.random.default_rng(0)
    out: List[Tuple[np.ndarray, float, str]] = []
    base_order = np.argsort(np.asarray(bit_cost, dtype=np.float32))
    seen: set[Tuple[int, ...]] = set()
    for S in support_sizes:
        S = int(min(max(1, S), code.n))
        for jp in range(max(1, int(jitter_passes))):
            if jp == 0:
                order = base_order
            else:
                noise = rng.normal(0.0, 0.02 + 0.02 * jp, size=code.n)
                order = np.argsort(bit_cost + noise)
            support = np.asarray(order[:S], dtype=np.int64)
            sol = gf2_solve(code.h[:, support], syn)
            if sol is None:
                continue
            idx = support[np.flatnonzero(sol)]
            key = tuple(sorted(int(i) for i in idx.tolist()))
            if not key or key in seen:
                continue
            seen.add(key)
            score = float(np.sum(bit_cost[list(key)])) + 0.15 * len(key)
            out.append((np.array(key, dtype=np.int32), score, f"osd_S{S}_j{jp}"))
    return out


def greedy_syndrome_repair_candidates(code: LDPCCode, base_hard: np.ndarray, bit_cost: np.ndarray,
                                       max_steps: int = 48, max_candidates: int = 4) -> List[Tuple[np.ndarray, float, str]]:
    syn = code.syndrome(base_hard).astype(np.uint8)
    if syn.sum() == 0:
        return []
    selected: List[int] = []
    out: List[Tuple[np.ndarray, float, str]] = []
    cur_syn = syn.copy()
    costs = np.asarray(bit_cost, dtype=np.float32)
    used = np.zeros(code.n, dtype=bool)
    for step in range(int(max_steps)):
        best_j = -1
        best_score = 1e9
        cur_w = int(cur_syn.sum())
        for j in np.argsort(costs)[: min(code.n, 256)]:
            j = int(j)
            if used[j]:
                continue
            new_syn = cur_syn ^ code.h[:, j]
            gain = cur_w - int(new_syn.sum())
            score = float(costs[j]) - 0.75 * gain
            if score < best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            break
        used[best_j] = True
        selected.append(best_j)
        cur_syn ^= code.h[:, best_j]
        if cur_syn.sum() == 0:
            arr = np.array(sorted(selected), dtype=np.int32)
            out.append((arr, float(np.sum(costs[arr])) + 0.2 * len(arr), "greedy"))
            if len(out) >= max_candidates:
                break
    return out


def build_candidate_bank(code: LDPCCode, llr_internal: np.ndarray, feature_pack: Dict[str, np.ndarray],
                         rescue_cfg: Dict, true_codeword: np.ndarray | None, max_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_hard = np.asarray(feature_pack["grand_base_hard"], dtype=np.uint8)
    bit_cost = np.asarray(feature_pack["bit_cost"], dtype=np.float32)
    feats = np.zeros((max_candidates, 14), dtype=np.float32)
    labels = np.zeros((max_candidates,), dtype=np.uint8)
    valid = np.zeros((max_candidates,), dtype=np.uint8)
    row = 0
    used: set[Tuple[int, ...]] = set()

    def try_add(mask_idx: Sequence[int], prior_score: float, source: str, force_label: bool = False) -> None:
        nonlocal row
        if row >= max_candidates:
            return
        key = tuple(sorted(set(int(i) for i in mask_idx)))
        if key in used:
            return
        used.add(key)
        mask = np.zeros(code.n, dtype=np.uint8)
        if key:
            mask[list(key)] = 1
        cand = (base_hard ^ mask).astype(np.uint8)
        if not code.is_codeword(cand):
            return
        crc_ok = code.crc_check_internal(cand)
        label = bool(force_label)
        if true_codeword is not None and np.array_equal(cand, np.asarray(true_codeword, dtype=np.uint8)):
            label = True
        feats[row] = candidate_feature_vector(code, llr_internal, base_hard, cand, mask, feature_pack, prior_score, crc_ok)
        labels[row] = np.uint8(label)
        valid[row] = 1
        row += 1

    if code.is_codeword(base_hard):
        try_add([], 0.0, "base_codeword")

    cands: List[Tuple[np.ndarray, float, str]] = []
    cands.extend(candidate_pool_from_feature_pack(
        feature_pack,
        max_weight=int(rescue_cfg.get("candidate_bank_max_weight", 12)),
        budget=int(rescue_cfg.get("candidate_bank_budget", 256)),
        pool_size=int(rescue_cfg.get("candidate_bank_pool_size", 160)),
        combo_pool_w_le3=int(rescue_cfg.get("combo_pool_w_le3", 28)),
        combo_pool_w_gt3=int(rescue_cfg.get("combo_pool_w_gt3", 18)),
    ))
    if bool(rescue_cfg.get("candidate_bank_enable_greedy", True)):
        cands.extend(greedy_syndrome_repair_candidates(
            code, base_hard, bit_cost,
            max_steps=int(rescue_cfg.get("candidate_bank_greedy_steps", 24)),
            max_candidates=int(rescue_cfg.get("candidate_bank_greedy_candidates", 4)),
        ))
    if bool(rescue_cfg.get("candidate_bank_enable_osd", True)):
        cands.extend(syndrome_osd_candidates(
            code, base_hard, bit_cost,
            support_sizes=rescue_cfg.get("candidate_bank_osd_support_sizes", [64, 96, 128]),
            jitter_passes=int(rescue_cfg.get("candidate_bank_osd_jitter_passes", 1)),
        ))
    cands = sorted(cands, key=lambda x: float(x[1]))
    for mask_idx, prior_score, source in cands:
        if row >= max_candidates:
            break
        try_add(mask_idx, prior_score, source)

    # Training-only oracle-positive injection. Evaluation never calls this path with true_codeword.
    if bool(rescue_cfg.get("candidate_bank_inject_oracle_positive", False)) and true_codeword is not None and not labels[:row].any():
        true_codeword = np.asarray(true_codeword, dtype=np.uint8).reshape(-1)
        oracle_mask = (base_hard ^ true_codeword).astype(np.uint8)
        weight = int(oracle_mask.sum())
        if 0 < weight <= int(rescue_cfg.get("oracle_candidate_max_weight", rescue_cfg.get("max_expanded_weight", 32))):
            idx = np.flatnonzero(oracle_mask)
            # Force insert even if not in generated candidate list; it is only a supervised reranker example.
            if row < max_candidates:
                mask = oracle_mask
                cand = true_codeword.copy()
                key = tuple(sorted(int(i) for i in idx.tolist()))
                if key not in used:
                    crc_ok = code.crc_check_internal(cand)
                    feats[row] = candidate_feature_vector(code, llr_internal, base_hard, cand, mask, feature_pack,
                                                          float(np.sum(bit_cost[idx])) + 0.15 * weight, crc_ok)
                    labels[row] = 1
                    valid[row] = 1
                    row += 1
    return feats, labels, valid


def build_training_labels(code: LDPCCode, feature_pack: Dict[str, np.ndarray], true_codeword: np.ndarray,
                          bp_result: BPDecodeResult, rescue_cfg: Dict, model_cfg: Dict,
                          llr_internal: np.ndarray) -> Dict[str, np.ndarray]:
    true_codeword = np.asarray(true_codeword, dtype=np.uint8).reshape(-1)
    base_hard = np.asarray(feature_pack["grand_base_hard"], dtype=np.uint8)
    target_mask = (base_hard ^ true_codeword).astype(np.uint8)
    target_idx = np.flatnonzero(target_mask)
    target_weight = int(target_idx.size)
    bp_residual_mask = (np.asarray(bp_result.hard, dtype=np.uint8) ^ true_codeword).astype(np.uint8)
    num_segments = int(model_cfg.get("num_segments", 8))
    max_weight_class = int(model_cfg.get("max_weight_class", 64))
    seg_labels = np.zeros((num_segments,), dtype=np.uint8)
    if target_idx.size:
        segs = feature_pack["segment_index"][target_idx]
        seg_labels[np.unique(segs)] = 1
    weight_label = min(target_weight, max_weight_class)
    reach_mode = str(rescue_cfg.get("reachability_label_mode", "weight_only")).lower()
    order = feature_pack["heuristic_order"]
    top_std = set(int(x) for x in order[: int(rescue_cfg.get("pool_size", 64))])
    top_exp = set(int(x) for x in order[: int(rescue_cfg.get("expanded_pool_size", 192))])
    if reach_mode in {"weight_only", "budget", "residual_weight"}:
        standard_reachable = int(0 < target_weight <= int(rescue_cfg.get("max_standard_weight", 10)))
        expanded_reachable = int(0 < target_weight <= int(rescue_cfg.get("max_expanded_weight", 32)))
    else:
        standard_reachable = int(0 < target_weight <= int(rescue_cfg.get("max_standard_weight", 10)) and all(int(i) in top_std for i in target_idx))
        expanded_reachable = int(0 < target_weight <= int(rescue_cfg.get("max_expanded_weight", 32)) and all(int(i) in top_exp for i in target_idx))
    rescueable = int((not standard_reachable) and expanded_reachable)
    candidate_feats, candidate_labels, candidate_valid = build_candidate_bank(
        code=code,
        llr_internal=llr_internal,
        feature_pack=feature_pack,
        rescue_cfg=rescue_cfg,
        true_codeword=true_codeword,
        max_candidates=int(model_cfg.get("rerank_list_size", 12)),
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
    }
