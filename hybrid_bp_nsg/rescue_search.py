from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .codes.gf2 import gf2_solve_support
from .codes.peg_ldpc import LDPCCode


def channel_metric_internal(llr_internal: np.ndarray, codeword_internal: np.ndarray) -> float:
    bipolar = 1.0 - 2.0 * np.asarray(codeword_internal, dtype=np.float32)
    return float(np.sum(np.asarray(llr_internal, dtype=np.float32) * bipolar))


def candidate_feature_vector(
    code: LDPCCode,
    llr_internal: np.ndarray,
    base_hard: np.ndarray,
    cand_codeword: np.ndarray,
    mask: np.ndarray,
    feature_pack: Dict[str, np.ndarray],
    prior_score: float,
    crc_ok: bool,
) -> np.ndarray:
    idx = np.flatnonzero(mask)
    vf = feature_pack["var_features"]
    susp = np.asarray(feature_pack.get("suspicion", np.zeros(code.n, dtype=np.float32)), dtype=np.float32)
    if idx.size == 0:
        local = np.zeros((1, vf.shape[1]), dtype=np.float32)
        local_susp = np.zeros((1,), dtype=np.float32)
    else:
        local = vf[idx]
        local_susp = susp[idx]
    cm = channel_metric_internal(llr_internal, cand_codeword)
    base_cm = channel_metric_internal(llr_internal, base_hard)
    return np.array([
        float(idx.size),
        float(np.sum(local_susp)),
        float(np.mean(1.0 - np.clip(local[:, 3], 0.0, 1.0))),
        float(np.mean(local[:, 11])) if local.shape[1] > 11 else 0.0,
        float(np.mean(local[:, 5])) if local.shape[1] > 5 else 0.0,
        float(np.mean(local[:, 7])) if local.shape[1] > 7 else 0.0,
        float(prior_score),
        float(cm / max(1, code.n)),
        float((cm - base_cm) / max(1, code.n)),
        float(code.is_codeword(cand_codeword)),
        float(1.0 if crc_ok else 0.0),
        float(np.mean(local[:, 14])) if local.shape[1] > 14 else 0.0,
        float(idx.size / max(1, code.n)),
        float(np.mean(local[:, 13])) if local.shape[1] > 13 else 0.0,
    ], dtype=np.float32)


def candidate_pool_from_feature_pack(
    feature_pack: Dict[str, np.ndarray],
    pool_size: int,
    top_k_bits: int,
    top_k_unsat: int,
    top_k_oscillation: int,
    bit_cost: np.ndarray | None = None,
) -> List[int]:
    vf = feature_pack["var_features"]
    heuristic_order = feature_pack["heuristic_order"].tolist()
    by_unsat = np.argsort(-(vf[:, 5] + vf[:, 11]))[:top_k_unsat].tolist()
    by_osc = np.argsort(-vf[:, 7])[:top_k_oscillation].tolist()
    by_component = np.argsort(-vf[:, -1])[:max(4, top_k_unsat)].tolist()
    by_logits_hint = heuristic_order[:top_k_bits]
    by_policy = np.argsort(bit_cost)[:min(len(vf), max(top_k_bits * 2, pool_size))].tolist() if bit_cost is not None else []
    pool: List[int] = []
    seen = set()
    for seq in [by_policy, by_logits_hint, by_component, by_unsat, by_osc, heuristic_order[:pool_size]]:
        for idx in seq:
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                pool.append(idx)
    return pool[: max(pool_size, top_k_bits)]


def component_templates(components: Sequence[np.ndarray], bit_cost: np.ndarray, max_weight: int) -> List[np.ndarray]:
    templates: List[np.ndarray] = []
    for comp in components[:8]:
        comp = np.asarray(comp, dtype=np.int32)
        if comp.size == 0:
            continue
        order = comp[np.argsort(bit_cost[comp])]
        templates.append(order[: min(order.size, max_weight)])
        if order.size > 2:
            templates.append(order[: min(max_weight, max(2, order.size // 2))])
        if order.size <= max_weight:
            templates.append(order)
    return templates


def enumerate_mask_candidates(
    pool: Sequence[int],
    components: Sequence[np.ndarray],
    bit_cost: np.ndarray,
    weight_candidates: Sequence[int],
    budget: int,
    weight_penalties: Sequence[float],
    combo_pool_w_le3: int = 24,
    combo_pool_w_gt3: int = 16,
    component_bonus: float = -0.25,
) -> List[Tuple[np.ndarray, float, str]]:
    scored: List[Tuple[np.ndarray, float, str]] = []
    seen = set()
    pool = list(dict.fromkeys(int(x) for x in pool))
    max_w = max([int(w) for w in weight_candidates], default=1)
    templates = component_templates(components, bit_cost, max_w)

    def add(mask_idx: Sequence[int], source: str, extra_bonus: float = 0.0) -> None:
        idx = tuple(sorted(set(int(i) for i in mask_idx)))
        if not idx or idx in seen:
            return
        seen.add(idx)
        w = len(idx)
        score = float(np.sum(bit_cost[list(idx)]))
        if w < len(weight_penalties):
            score += float(weight_penalties[w])
        else:
            score += float(weight_penalties[-1] + 0.40 * (w - len(weight_penalties) + 1))
        score += float(extra_bonus)
        scored.append((np.array(idx, dtype=np.int32), score, source))

    for tmpl in templates:
        add(tmpl, "component", component_bonus)

    for w in [int(x) for x in weight_candidates]:
        combo_pool = pool[: min(len(pool), combo_pool_w_le3 if w <= 3 else combo_pool_w_gt3)]
        if w <= 0 or w > len(combo_pool):
            continue
        count = 0
        if w <= 5:
            for combo in combinations(combo_pool, w):
                add(combo, "combo")
                count += 1
                if count >= budget * 2:
                    break
        for tmpl in templates[:10]:
            if tmpl.size == 0:
                continue
            if tmpl.size == w:
                add(tmpl, "component_exact", component_bonus)
            elif tmpl.size < w:
                s = set(int(i) for i in tmpl)
                extras = [x for x in combo_pool if x not in s]
                need = w - tmpl.size
                if len(extras) >= need:
                    add(list(tmpl) + extras[:need], "component_plus", component_bonus)
            else:
                add(tmpl[:w], "component_trim", component_bonus)

    scored.sort(key=lambda x: x[1])
    return scored[:budget]


def syndrome_osd_candidates(
    code: LDPCCode,
    base_hard: np.ndarray,
    bit_cost: np.ndarray,
    support_sizes: Sequence[int],
    jitter_passes: int = 1,
) -> List[Tuple[np.ndarray, float, str]]:
    syndrome = code.syndrome(base_hard)
    if int(syndrome.sum()) == 0:
        return []
    n = code.n
    order = np.argsort(bit_cost)
    candidates: List[Tuple[np.ndarray, float, str]] = []
    seen = set()

    def add_from_support(support: np.ndarray, tag: str) -> None:
        support = np.asarray(support, dtype=np.int64)
        support = support[(support >= 0) & (support < n)]
        support = np.array(list(dict.fromkeys(int(x) for x in support)), dtype=np.int64)
        if support.size == 0:
            return
        sol = gf2_solve_support(code.h, support, syndrome)
        if sol is None:
            return
        loc = support[np.flatnonzero(sol)]
        if loc.size == 0:
            return
        key = tuple(sorted(int(x) for x in loc))
        if key in seen:
            return
        seen.add(key)
        score = float(np.sum(bit_cost[list(key)]) + 0.02 * len(key))
        candidates.append((np.array(key, dtype=np.int32), score, tag))

    for size in support_sizes:
        size = max(1, min(int(size), n))
        add_from_support(order[:size], f"osd_{size}")

    if jitter_passes:
        base_syn_vars = np.flatnonzero((syndrome.astype(np.float32) @ code.h.astype(np.float32)) > 0)
        if base_syn_vars.size:
            syn_order = base_syn_vars[np.argsort(bit_cost[base_syn_vars])]
            for j in range(int(jitter_passes)):
                for size in support_sizes:
                    size = max(1, min(int(size), n))
                    head = syn_order[: min(len(syn_order), max(8, size // 4 + 4))]
                    head_set = set(int(v) for v in head)
                    tail = [x for x in order if x not in head_set]
                    support = np.array(list(head) + tail[: max(0, size - len(head))], dtype=np.int64)
                    add_from_support(support, f"osd_syn{j}_{size}")

    candidates.sort(key=lambda x: x[1])
    return candidates


def greedy_syndrome_repair_candidates(
    code: LDPCCode,
    base_hard: np.ndarray,
    bit_cost: np.ndarray,
    max_steps: int,
    max_candidates: int,
) -> List[Tuple[np.ndarray, float, str]]:
    syn0 = code.syndrome(base_hard).astype(np.uint8)
    if int(syn0.sum()) == 0:
        return []
    h = code.h
    out: List[Tuple[np.ndarray, float, str]] = []
    for seed in range(max_candidates):
        syn = syn0.copy()
        mask = np.zeros(code.n, dtype=np.uint8)
        tabu = set()
        for step in range(max_steps):
            unsat_part = syn.astype(np.float32) @ h.astype(np.float32)
            sat_part = (1 - syn).astype(np.float32) @ h.astype(np.float32)
            gain = unsat_part - 0.55 * sat_part - 0.15 * bit_cost
            if seed:
                gain -= 0.01 * ((np.arange(code.n) * (seed + 3)) % 17)
            for t in tabu:
                gain[int(t)] = -1e9
            v = int(np.argmax(gain))
            if gain[v] <= 0 and step > 0:
                break
            tabu.add(v)
            mask[v] ^= 1
            syn ^= h[:, v]
            if int(syn.sum()) == 0:
                idx = np.flatnonzero(mask)
                if idx.size:
                    score = float(np.sum(bit_cost[idx]) + 0.05 * idx.size)
                    out.append((idx.astype(np.int32), score, f"greedy_{seed}"))
                break
    dedup = {}
    for idx, score, src in out:
        key = tuple(sorted(int(x) for x in idx))
        if key not in dedup or score < dedup[key][0]:
            dedup[key] = (score, src)
    ans = [(np.array(key, dtype=np.int32), score, src) for key, (score, src) in dedup.items()]
    ans.sort(key=lambda x: x[1])
    return ans
