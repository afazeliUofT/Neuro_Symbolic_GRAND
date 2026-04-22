from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from ..codes.gf2 import gf2_solve_support
from ..codes.peg_ldpc import LDPCCode
from ..decoders.bp import BeliefPropagationDecoder, BPDecodeResult
from ..training.features import build_rescue_features, make_grand_base


@dataclass
class RescueResult:
    success: bool
    final_codeword: np.ndarray
    final_hard: np.ndarray
    queries: int
    elapsed_ms: float
    action: str
    used_micro_bp: bool
    rerank_candidates: int
    packet_score: float
    rescue_invoked: bool
    main_result: BPDecodeResult


@dataclass
class ValidCandidate:
    mask: np.ndarray
    codeword: np.ndarray
    score: float
    source: str
    used_micro_bp: bool
    cand_features: np.ndarray


def _channel_metric(llr: np.ndarray, codeword: np.ndarray) -> float:
    bipolar = 1.0 - 2.0 * codeword.astype(np.float32)
    return float(np.sum(llr.astype(np.float32) * bipolar))


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


def _candidate_pool_from_feature_pack(
    feature_pack: Dict[str, np.ndarray],
    pool_size: int,
    top_k_bits: int,
    top_k_unsat: int,
    top_k_oscillation: int,
    bit_cost: Optional[np.ndarray] = None,
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


def _component_templates(components: Sequence[np.ndarray], bit_cost: np.ndarray, max_weight: int) -> List[np.ndarray]:
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


def _enumerate_candidates(
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
    templates = _component_templates(components, bit_cost, max_w)

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
        # Exact combinations are only sensible for low weights. For higher weights, the
        # Tanner-syndrome OSD path is responsible for producing candidates.
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


def _syndrome_osd_candidates(
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

    # Deterministic jitter passes: move some syndrome-participating bits earlier.
    if jitter_passes:
        base_syn_vars = np.flatnonzero((syndrome.astype(np.float32) @ code.h.astype(np.float32)) > 0)
        if base_syn_vars.size:
            syn_order = base_syn_vars[np.argsort(bit_cost[base_syn_vars])]
            for j in range(int(jitter_passes)):
                for size in support_sizes:
                    size = max(1, min(int(size), n))
                    head = syn_order[: min(len(syn_order), max(8, size // 4 + 4))]
                    tail = [x for x in order if x not in set(int(v) for v in head)]
                    support = np.array(list(head) + tail[: max(0, size - len(head))], dtype=np.int64)
                    add_from_support(support, f"osd_syn{j}_{size}")

    candidates.sort(key=lambda x: x[1])
    return candidates


def _greedy_syndrome_repair_candidates(
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
                # Deterministic perturbation to generate a small family of repairs.
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
    # De-duplicate.
    dedup = {}
    for idx, score, src in out:
        key = tuple(sorted(int(x) for x in idx))
        if key not in dedup or score < dedup[key][0]:
            dedup[key] = (score, src)
    ans = [(np.array(key, dtype=np.int32), score, src) for key, (score, src) in dedup.items()]
    ans.sort(key=lambda x: x[1])
    return ans


class ResidualGrandRescueDecoder:
    def __init__(
        self,
        code: LDPCCode,
        main_bp: BeliefPropagationDecoder,
        micro_bp: BeliefPropagationDecoder,
        rescue_cfg: Dict[str, object],
        mode: str = "orb",
        rescue_net=None,
        device: str = "cpu",
    ):
        self.code = code
        self.main_bp = main_bp
        self.micro_bp = micro_bp
        self.cfg = rescue_cfg
        self.mode = str(mode)
        self.rescue_net = rescue_net
        self.device = str(device or ("/GPU:0" if tf is not None and tf.config.list_physical_devices("GPU") else "/CPU:0"))

    def _heuristic_policy(self, llr: np.ndarray, feature_pack: Dict[str, np.ndarray]) -> Dict[str, object]:
        vf = feature_pack["var_features"]
        abs_llr = np.abs(llr)
        rank_cost = np.linspace(0.0, 1.0, self.code.n, endpoint=False, dtype=np.float32)
        inv_cost = np.empty_like(rank_cost)
        inv_cost[feature_pack["heuristic_order"]] = rank_cost

        # Low channel |LLR| and high suspicion should have low cost.
        channel_unreliability_bonus = -0.25 * (1.0 - np.clip(vf[:, 3], 0.0, 1.0))
        syndrome_bonus = -0.18 * np.clip(vf[:, 11], 0.0, 1.0)
        disagreement_bonus = -0.10 * vf[:, 13]
        puncture_bonus = -0.05 * vf[:, 14]

        if self.mode in {"cdf", "segmented", "ai"}:
            bit_cost = np.log1p(inv_cost * 10.0)
        else:
            bit_cost = inv_cost.copy()
        bit_cost = bit_cost + 0.03 * abs_llr / max(1e-6, float(abs_llr.max())) + channel_unreliability_bonus + syndrome_bonus + disagreement_bonus + puncture_bonus

        if self.mode == "orb":
            weight_candidates = list(range(1, min(6, int(self.cfg["max_standard_weight"])) + 1))
            pool_size = min(int(self.cfg["pool_size"]), max(16, int(self.cfg.get("top_k_bits", 64))))
            budget = min(int(self.cfg["standard_budget"]), int(self.cfg.get("direct_budget", self.cfg["standard_budget"])))
            gate = 1.0
            expanded = 0.0
        else:
            max_exp = int(self.cfg["max_expanded_weight"])
            weight_candidates = list(range(1, min(max_exp, 10) + 1))
            pool_size = int(self.cfg["expanded_pool_size"])
            budget = int(self.cfg["expanded_budget"])
            gate = 1.0
            expanded = 1.0

        return {
            "bit_cost": bit_cost.astype(np.float32),
            "weight_candidates": weight_candidates,
            "pool_size": pool_size,
            "budget": budget,
            "gate": gate,
            "expanded": expanded,
            "rescue": 1.0,
            "packet_emb": None,
            "candidate_net": None,
        }

    def _ai_policy(self, llr: np.ndarray, feature_pack: Dict[str, np.ndarray], snr_db: float, profile: str) -> Dict[str, object]:
        if self.rescue_net is None or tf is None:
            return self._heuristic_policy(llr, feature_pack)

        with tf.device(self.device):
            batch = {
                "var_features": tf.convert_to_tensor(feature_pack["var_features"][None, ...], dtype=tf.float32),
                "check_features": tf.convert_to_tensor(feature_pack["check_features"][None, ...], dtype=tf.float32),
                "global_features": tf.convert_to_tensor(feature_pack["global_features"][None, ...], dtype=tf.float32),
                "heuristic_order": tf.convert_to_tensor(feature_pack["heuristic_order"][None, ...], dtype=tf.int32),
            }
            out = self.rescue_net(batch, training=False)
            bit_prob = tf.sigmoid(tf.cast(out["bit_logits"], tf.float32)).numpy()[0]
            seg_prob = tf.sigmoid(tf.cast(out["segment_logits"], tf.float32)).numpy()[0]
            std_p = float(tf.sigmoid(tf.cast(out["standard_logits"], tf.float32)).numpy()[0])
            exp_p = float(tf.sigmoid(tf.cast(out["expanded_logits"], tf.float32)).numpy()[0])
            rescue_p = float(tf.sigmoid(tf.cast(out["rescue_logits"], tf.float32)).numpy()[0])
            weight_prob = tf.nn.softmax(tf.cast(out["weight_logits"], tf.float32), axis=-1).numpy()[0]
            packet_emb = out["packet_embedding"]

        vf = feature_pack["var_features"]
        seg_idx = feature_pack["segment_index"]
        bit_cost = -np.log(np.clip(bit_prob, 1e-5, 1 - 1e-5))

        # v11 bug fix: inverse-rank per variable, not double-argsort(order).
        rank_cost = np.linspace(0.0, 1.0, self.code.n, endpoint=False, dtype=np.float32)
        inv_cost = np.empty_like(rank_cost)
        inv_cost[feature_pack["heuristic_order"]] = rank_cost
        bit_cost += 0.08 * inv_cost

        bit_cost -= float(self.cfg.get("segment_bonus_scale", 0.0)) * seg_prob[seg_idx]
        bit_cost -= float(self.cfg.get("component_focus_scale", 0.0)) * vf[:, -1]
        bit_cost -= float(self.cfg.get("oscillation_focus_scale", 0.0)) * vf[:, 7]
        bit_cost -= 0.10 * vf[:, 11]  # base syndrome participation
        bit_cost -= 0.06 * vf[:, 13]  # BP/channel disagreement

        likely_weights = np.argsort(-weight_prob)[: int(self.cfg.get("likely_weight_topk", 8))].tolist()
        likely_weights = [int(w) for w in likely_weights if int(w) > 0]
        if not likely_weights:
            likely_weights = [1, 2, 3, 4]

        max_std = int(self.cfg["max_standard_weight"])
        max_exp = int(self.cfg["max_expanded_weight"])
        standard = [min(max(w, 1), max_std) for w in likely_weights if w <= max_std + 1]
        expanded = [min(max(w, 1), max_exp) for w in likely_weights if w <= max_exp + 2]
        if rescue_p >= float(self.cfg.get("rescue_threshold", 0.05)) or exp_p >= float(self.cfg.get("expanded_threshold", 0.2)):
            expanded = sorted(set(expanded + list(range(1, min(max_exp, 10) + 1))))
        standard = sorted(set(standard)) or list(range(1, min(max_std, 6) + 1))
        expanded = sorted(set(expanded + standard)) or standard
        use_expanded = (
            bool(self.cfg.get("always_rescue_after_bp_fail", True))
            or exp_p >= float(self.cfg.get("expanded_threshold", 0.2))
            or rescue_p >= float(self.cfg.get("rescue_threshold", 0.05))
        )

        return {
            "bit_cost": bit_cost.astype(np.float32),
            "weight_candidates": expanded if use_expanded else standard,
            "pool_size": int(self.cfg["expanded_pool_size"] if use_expanded else self.cfg["pool_size"]),
            "budget": int(self.cfg["expanded_budget"] if use_expanded else self.cfg["standard_budget"]),
            "gate": std_p,
            "expanded": exp_p,
            "rescue": rescue_p,
            "packet_emb": packet_emb,
            "candidate_net": self.rescue_net.reranker if bool(self.cfg.get("rerank_use_net", True)) else None,
        }

    def _run_search_on_base(
        self,
        llr: np.ndarray,
        base_hard: np.ndarray,
        main_result: BPDecodeResult,
        feature_pack: Dict[str, np.ndarray],
        policy: Dict[str, object],
        rescue_p: float,
    ) -> Tuple[List[ValidCandidate], int, bool]:
        bit_cost = np.asarray(policy["bit_cost"], dtype=np.float32)
        pool = _candidate_pool_from_feature_pack(
            feature_pack,
            int(policy["pool_size"]),
            int(self.cfg["top_k_bits"]),
            int(self.cfg["top_k_unsat"]),
            int(self.cfg["top_k_oscillation"]),
            bit_cost=bit_cost,
        )
        candidates = _enumerate_candidates(
            pool=pool,
            components=feature_pack["components"],
            bit_cost=bit_cost,
            weight_candidates=policy["weight_candidates"],
            budget=int(policy["budget"]),
            weight_penalties=self.cfg["weight_penalties"],
            combo_pool_w_le3=int(self.cfg.get("combo_pool_w_le3", 24)),
            combo_pool_w_gt3=int(self.cfg.get("combo_pool_w_gt3", 16)),
            component_bonus=float(self.cfg.get("component_bonus", -0.25)),
        )
        if bool(self.cfg.get("enable_greedy_repair", True)):
            candidates.extend(_greedy_syndrome_repair_candidates(
                self.code, base_hard, bit_cost,
                max_steps=int(self.cfg.get("greedy_repair_steps", 32)),
                max_candidates=int(self.cfg.get("greedy_repair_candidates", 6)),
            ))
        if bool(self.cfg.get("enable_osd_repair", True)):
            candidates.extend(_syndrome_osd_candidates(
                self.code, base_hard, bit_cost,
                support_sizes=self.cfg.get("osd_support_sizes", [96, 128, 192, 256]),
                jitter_passes=int(self.cfg.get("osd_jitter_passes", 1)),
            ))
        candidates.sort(key=lambda x: x[1])

        valid: List[ValidCandidate] = []
        queries = 0
        used_micro = False
        failed_for_micro: List[tuple[np.ndarray, float, str, np.ndarray]] = []
        direct_budget = min(int(policy["budget"]), int(self.cfg.get("direct_budget", policy["budget"])))

        # If the base itself is a codeword, consider it.
        if int(self.code.syndrome(base_hard).sum()) == 0:
            mask = np.zeros(self.code.n, dtype=np.uint8)
            valid.append(ValidCandidate(mask, base_hard.copy(), 0.0, "base_codeword", False, _mask_features(mask, feature_pack)))

        seen = set()
        for mask_idx, prior_score, source in candidates:
            key = tuple(sorted(int(x) for x in mask_idx))
            if key in seen:
                continue
            seen.add(key)
            queries += 1
            mask = np.zeros(self.code.n, dtype=np.uint8)
            mask[list(key)] = 1
            cand = base_hard.copy()
            cand[list(key)] ^= 1
            syn = self.code.syndrome(cand)
            cand_feat = _mask_features(mask, feature_pack)
            if int(syn.sum()) == 0:
                valid.append(ValidCandidate(mask, cand.copy(), float(prior_score), source, False, cand_feat))
            elif self.mode == "ai":
                failed_for_micro.append((mask.copy(), float(prior_score), source, cand_feat))
            if len(valid) >= int(self.cfg.get("rerank_list_size", 8)) and queries >= int(self.cfg.get("rerank_extra_queries", 32)):
                break
            if queries >= direct_budget:
                break

        if (not valid) and bool(self.cfg.get("enable_micro_bp", True)) and self.mode == "ai":
            micro_thr = float(self.cfg.get("micro_trigger_threshold", self.cfg.get("rescue_threshold", 0.05)))
            micro_cap = int(self.cfg.get("micro_candidate_cap", 4))
            if rescue_p >= micro_thr and failed_for_micro:
                shortlist = failed_for_micro
                if policy.get("candidate_net") is not None and policy.get("packet_emb") is not None and tf is not None:
                    cand_feats_np = np.stack([x[3] for x in failed_for_micro], axis=0).astype(np.float32)
                    with tf.device(self.device):
                        cand_feats = tf.convert_to_tensor(cand_feats_np[None, ...], dtype=tf.float32)
                        packet_emb = policy["packet_emb"]
                        micro_scores = policy["candidate_net"](packet_emb, cand_feats, training=False).numpy()[0]
                    order = np.argsort(-(micro_scores + float(self.cfg.get("rerank_prior_scale", 0.10)) * np.array([-x[1] for x in failed_for_micro], dtype=np.float32)))
                    shortlist = [failed_for_micro[i] for i in order[:micro_cap]]
                else:
                    shortlist = failed_for_micro[:micro_cap]
                for mask, prior_score, source, cand_feat in shortlist:
                    mask_idx = np.flatnonzero(mask)
                    # Micro-BP is still initialized around the channel LLR, but nudged according
                    # to the candidate mask relative to the GRAND base.
                    llr_mod = np.asarray(llr, dtype=np.float32).copy()
                    llr_mod[mask_idx] *= -float(self.cfg.get("micro_flip_scale", 1.10))
                    micro = self.micro_bp.decode(llr_mod, max_iters=self.micro_bp.max_iters, collect_trace=False)
                    if micro.success:
                        used_micro = True
                        valid.append(ValidCandidate(mask, micro.hard.copy(), float(prior_score), source + "_micro", True, cand_feat))
                    if len(valid) >= int(self.cfg.get("rerank_list_size", 8)):
                        break
        return valid, queries, used_micro

    def rescue(self, llr: np.ndarray, main_result: BPDecodeResult, snr_db: float, profile: str) -> RescueResult:
        start = time.perf_counter()
        if main_result.success:
            return RescueResult(True, main_result.hard.copy(), main_result.hard.copy(), 0,
                                (time.perf_counter() - start) * 1e3, "bp_success", False, 0, 1.0, False, main_result)

        target_basis = str(self.cfg.get("target_basis", "channel_with_bp_punctures"))
        feature_pack = build_rescue_features(
            self.code, llr, main_result, snr_db, profile,
            num_segments=int(self.cfg.get("num_segments", 8)) if "num_segments" in self.cfg else 8,
            target_basis=target_basis,
        )
        if self.mode == "ai":
            policy = self._ai_policy(llr, feature_pack, snr_db, profile)
        else:
            policy = self._heuristic_policy(llr, feature_pack)

        gate = float(policy["gate"])
        rescue_p = float(policy.get("rescue", 0.0))
        if self.mode == "ai" and not bool(self.cfg.get("always_rescue_after_bp_fail", True)):
            hopeless_thr = float(self.cfg.get("hopeless_threshold", -1.0))
            rescue_thr = float(self.cfg.get("rescue_threshold", 0.05))
            gate_thr = float(self.cfg.get("gating_threshold", -1.0))
            expanded_thr = float(self.cfg.get("expanded_threshold", 0.2))
            should_rescue = (gate >= gate_thr) or (rescue_p >= rescue_thr) or (float(policy.get("expanded", 0.0)) >= expanded_thr)
            if (gate < hopeless_thr and rescue_p < rescue_thr) or (not should_rescue):
                return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), 0,
                                    (time.perf_counter() - start) * 1e3, "skip_hopeless", False, 0, gate, False, main_result)

        base_hard = np.asarray(feature_pack["grand_base_hard"], dtype=np.uint8)
        valid, queries, used_micro = self._run_search_on_base(llr, base_hard, main_result, feature_pack, policy, rescue_p)

        # Optional fallback: if channel-basis rescue did not produce any valid candidate,
        # try failed-BP basis. This preserves the old route but no longer makes it primary.
        if (not valid) and bool(self.cfg.get("try_bp_basis_fallback", True)):
            fp2 = build_rescue_features(
                self.code, llr, main_result, snr_db, profile,
                num_segments=int(self.cfg.get("num_segments", 8)) if "num_segments" in self.cfg else 8,
                target_basis="bp",
            )
            # Reuse bit_cost from policy but feature pools from BP-basis pack.
            valid2, q2, micro2 = self._run_search_on_base(llr, np.asarray(main_result.hard, dtype=np.uint8), main_result, fp2, policy, rescue_p)
            valid.extend(valid2)
            queries += q2
            used_micro = used_micro or micro2

        elapsed_ms = (time.perf_counter() - start) * 1e3
        if not valid:
            return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), queries, elapsed_ms,
                                "rescue_fail", used_micro, 0, gate, True, main_result)

        if policy.get("candidate_net") is not None and policy.get("packet_emb") is not None and tf is not None:
            with tf.device(self.device):
                cand_feats = tf.convert_to_tensor(np.stack([v.cand_features for v in valid], axis=0)[None, ...], dtype=tf.float32)
                packet_emb = policy["packet_emb"]
                scores = policy["candidate_net"](packet_emb, cand_feats, training=False).numpy()[0]
            final_scores = (
                scores
                + float(self.cfg.get("rerank_prior_scale", 0.10)) * np.array([-v.score for v in valid], dtype=np.float32)
                + np.array([_channel_metric(llr, v.codeword) for v in valid], dtype=np.float32) / max(1, self.code.n)
            )
            best_idx = int(np.argmax(final_scores))
        else:
            metrics = np.array([_channel_metric(llr, v.codeword) - 0.10 * v.score for v in valid], dtype=np.float32)
            best_idx = int(np.argmax(metrics))

        best = valid[best_idx]
        action = "rescue_success_micro" if best.used_micro_bp else f"rescue_success_{best.source}"
        return RescueResult(True, best.codeword.copy(), best.codeword.copy(), queries, elapsed_ms,
                            action, best.used_micro_bp, len(valid), gate, True, main_result)
