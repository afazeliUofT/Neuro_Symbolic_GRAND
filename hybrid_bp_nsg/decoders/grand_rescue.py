from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple
import math
import time
import numpy as np
import torch

from ..codes.peg_ldpc import LDPCCode
from ..decoders.bp import BeliefPropagationDecoder, BPDecodeResult
from ..training.features import build_rescue_features


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


def _candidate_pool_from_feature_pack(feature_pack: Dict[str, np.ndarray], pool_size: int, top_k_bits: int,
                                      top_k_unsat: int, top_k_oscillation: int, bit_cost: Optional[np.ndarray] = None) -> List[int]:
    vf = feature_pack["var_features"]
    heuristic_order = feature_pack["heuristic_order"].tolist()
    by_unsat = np.argsort(-vf[:, 5])[:top_k_unsat].tolist()
    by_osc = np.argsort(-vf[:, 7])[:top_k_oscillation].tolist()
    by_component = np.argsort(-vf[:, -1])[:max(4, top_k_unsat)].tolist()
    by_logits_hint = heuristic_order[:top_k_bits]
    by_policy = np.argsort(bit_cost)[:min(len(vf), max(top_k_bits * 2, pool_size))].tolist() if bit_cost is not None else []
    pool = []
    seen = set()
    for seq in [by_policy, by_logits_hint, by_component, by_unsat, by_osc, heuristic_order[:pool_size]]:
        for idx in seq:
            if int(idx) not in seen:
                seen.add(int(idx))
                pool.append(int(idx))
    return pool[: max(pool_size, top_k_bits)]


def _component_templates(components: Sequence[np.ndarray], bit_cost: np.ndarray, max_weight: int) -> List[np.ndarray]:
    templates: List[np.ndarray] = []
    for comp in components[:6]:
        comp = np.asarray(comp, dtype=np.int16)
        if comp.size == 0:
            continue
        order = comp[np.argsort(bit_cost[comp])]
        templates.append(order[: min(order.size, max_weight)])
        if order.size > 2:
            templates.append(order[: min(max_weight, max(2, order.size // 2))])
        if order.size <= max_weight:
            templates.append(order)
    return templates


def _enumerate_candidates(pool: Sequence[int], components: Sequence[np.ndarray], bit_cost: np.ndarray,
                          weight_candidates: Sequence[int], budget: int, weight_penalties: Sequence[float],
                          component_bonus: float = -0.25) -> List[Tuple[np.ndarray, float, str]]:
    scored: List[Tuple[np.ndarray, float, str]] = []
    seen = set()
    pool = list(dict.fromkeys(int(x) for x in pool))
    templates = _component_templates(components, bit_cost, max(weight_candidates) if weight_candidates else 1)

    def add(mask_idx: Sequence[int], source: str, extra_bonus: float = 0.0) -> None:
        idx = tuple(sorted(int(i) for i in mask_idx))
        if not idx or idx in seen:
            return
        seen.add(idx)
        score = float(np.sum(bit_cost[list(idx)]))
        w = len(idx)
        if w < len(weight_penalties):
            score += float(weight_penalties[w])
        else:
            score += float(weight_penalties[-1] + 0.35 * (w - len(weight_penalties) + 1))
        score += extra_bonus
        scored.append((np.array(idx, dtype=np.int16), score, source))

    for tmpl in templates:
        add(tmpl, "component", component_bonus)

    for w in weight_candidates:
        combo_pool = pool[: min(len(pool), 18 if w <= 3 else 12)]
        if w <= 0 or w > len(combo_pool):
            continue
        count = 0
        for combo in combinations(combo_pool, w):
            add(combo, "combo")
            count += 1
            if count >= budget * 2:
                break
        for tmpl in templates[:8]:
            if tmpl.size == 0:
                continue
            if tmpl.size == w:
                add(tmpl, "component_exact", component_bonus)
            elif tmpl.size < w:
                extras = [x for x in combo_pool if x not in set(int(i) for i in tmpl)]
                need = w - tmpl.size
                if len(extras) >= need:
                    add(list(tmpl) + extras[:need], "component_plus", component_bonus)
            else:
                add(tmpl[:w], "component_trim", component_bonus)
    scored.sort(key=lambda x: x[1])
    return scored[:budget]


class ResidualGrandRescueDecoder:
    def __init__(self, code: LDPCCode, main_bp: BeliefPropagationDecoder, micro_bp: BeliefPropagationDecoder,
                 rescue_cfg: Dict[str, object], mode: str = "orb", rescue_net=None, device: str = "cpu"):
        self.code = code
        self.main_bp = main_bp
        self.micro_bp = micro_bp
        self.cfg = rescue_cfg
        self.mode = mode
        self.rescue_net = rescue_net
        self.device = torch.device(device)
        self._h_dense_torch = None
        self._deg_v_torch = None
        self._deg_c_torch = None

    def _torch_graph_buffers(self):
        if self._h_dense_torch is None:
            self._h_dense_torch = torch.tensor(self.code.h.astype(np.float32), dtype=torch.float32, device=self.device)
            self._deg_v_torch = torch.tensor(np.maximum(self.code.deg_v.astype(np.float32), 1.0), dtype=torch.float32, device=self.device)
            self._deg_c_torch = torch.tensor(np.maximum(self.code.deg_c.astype(np.float32), 1.0), dtype=torch.float32, device=self.device)
        return self._h_dense_torch, self._deg_v_torch, self._deg_c_torch

    def _heuristic_policy(self, llr: np.ndarray, feature_pack: Dict[str, np.ndarray]) -> Dict[str, object]:
        vf = feature_pack["var_features"]
        abs_llr = np.abs(llr)
        rank_cost = np.linspace(0.0, 1.0, self.code.n, endpoint=False, dtype=np.float32)
        inv_cost = np.empty_like(rank_cost)
        inv_cost[feature_pack["heuristic_order"]] = rank_cost
        if self.mode == "orb":
            bit_cost = inv_cost + 0.08 * abs_llr / max(1e-6, abs_llr.max())
            weight_candidates = [1, 2, 3, 4]
            pool_size = min(int(self.cfg["pool_size"]), 16)
            budget = min(int(self.cfg["standard_budget"]), 80)
            gate = 1.0
            expanded = 0.0
        elif self.mode == "cdf":
            bit_cost = np.log1p(inv_cost * 15.0) + 0.04 * abs_llr / max(1e-6, abs_llr.max())
            weight_candidates = [1, 2, 3, 4, 5]
            pool_size = min(int(self.cfg["expanded_pool_size"]), 20)
            budget = min(int(self.cfg["expanded_budget"]), 140)
            gate = 1.0
            expanded = 1.0
        elif self.mode == "segmented":
            seg = feature_pack["segment_index"]
            bit_cost = inv_cost + 0.25 * seg.astype(np.float32) / max(1, seg.max()) - 0.15 * vf[:, -1]
            weight_candidates = [1, 2, 3, 4]
            pool_size = min(int(self.cfg["pool_size"]), 16)
            budget = min(int(self.cfg["standard_budget"]), 80)
            gate = 1.0
            expanded = 0.5
        else:
            bit_cost = inv_cost + 0.06 * abs_llr / max(1e-6, abs_llr.max())
            weight_candidates = [1, 2, 3, 4]
            pool_size = min(int(self.cfg["pool_size"]), 16)
            budget = min(int(self.cfg["standard_budget"]), 80)
            gate = 1.0
            expanded = 0.0
        return {
            "bit_cost": bit_cost.astype(np.float32),
            "weight_candidates": weight_candidates,
            "pool_size": pool_size,
            "budget": budget,
            "gate": gate,
            "expanded": expanded,
            "rescue": 0.0,
            "packet_emb": None,
            "candidate_net": None,
        }

    def _ai_policy(self, llr: np.ndarray, feature_pack: Dict[str, np.ndarray], snr_db: float, profile: str) -> Dict[str, object]:
        if self.rescue_net is None:
            return self._heuristic_policy(llr, feature_pack)
        self.rescue_net.eval()
        h_dense, deg_v, deg_c = self._torch_graph_buffers()
        with torch.no_grad():
            batch = {
                "var_features": torch.tensor(feature_pack["var_features"][None, ...], dtype=torch.float32, device=self.device),
                "check_features": torch.tensor(feature_pack["check_features"][None, ...], dtype=torch.float32, device=self.device),
                "global_features": torch.tensor(feature_pack["global_features"][None, ...], dtype=torch.float32, device=self.device),
                "heuristic_order": torch.tensor(feature_pack["heuristic_order"][None, ...], dtype=torch.long, device=self.device),
            }
            out = self.rescue_net(batch["var_features"], batch["check_features"], batch["global_features"],
                                  batch["heuristic_order"], h_dense, deg_v, deg_c)
            bit_prob = torch.sigmoid(out["bit_logits"]).cpu().numpy()[0]
            seg_prob = torch.sigmoid(out["segment_logits"]).cpu().numpy()[0]
            std_p = float(torch.sigmoid(out["standard_logits"]).cpu().numpy()[0])
            exp_p = float(torch.sigmoid(out["expanded_logits"]).cpu().numpy()[0])
            rescue_p = float(torch.sigmoid(out["rescue_logits"]).cpu().numpy()[0])
            weight_prob = torch.softmax(out["weight_logits"], dim=-1).cpu().numpy()[0]
            packet_emb = out["packet_embedding"].cpu()
        vf = feature_pack["var_features"]
        seg_idx = feature_pack["segment_index"]
        bit_cost = -np.log(np.clip(bit_prob, 1e-5, 1 - 1e-5))
        bit_cost += 0.08 * np.linspace(0, 1, self.code.n, endpoint=False, dtype=np.float32)[np.argsort(np.argsort(feature_pack["heuristic_order"]))]
        bit_cost -= float(self.cfg.get("segment_bonus_scale", 0.0)) * seg_prob[seg_idx]
        bit_cost -= float(self.cfg.get("component_focus_scale", 0.0)) * vf[:, -1]
        bit_cost -= float(self.cfg.get("oscillation_focus_scale", 0.0)) * vf[:, 7]
        likely_weights = np.argsort(-weight_prob)[: int(self.cfg.get("likely_weight_topk", 6))].tolist()
        likely_weights = [w for w in likely_weights if w > 0]
        if not likely_weights:
            likely_weights = [1, 2, 3, 4]
        max_std = int(self.cfg["max_standard_weight"])
        max_exp = int(self.cfg["max_expanded_weight"])
        standard = [min(max(w, 1), max_std) for w in likely_weights if w <= max_std + 1]
        expanded = [min(max(w, 1), max_exp) for w in likely_weights if w <= max_exp + 2]
        if rescue_p >= float(self.cfg.get("rescue_threshold", 0.25)) or exp_p >= float(self.cfg.get("expanded_threshold", 0.5)):
            expanded = sorted(set(expanded + list(range(1, min(max_exp, 6) + 1))))
        standard = sorted(set(standard)) or list(range(1, max_std + 1))
        expanded = sorted(set(expanded + standard)) or standard
        use_expanded = exp_p >= float(self.cfg.get("expanded_threshold", 0.45)) or rescue_p >= float(self.cfg.get("rescue_threshold", 0.20))
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

    def rescue(self, llr: np.ndarray, main_result: BPDecodeResult, snr_db: float, profile: str) -> RescueResult:
        start = time.perf_counter()
        if main_result.success:
            return RescueResult(True, main_result.hard.copy(), main_result.hard.copy(), 0,
                                (time.perf_counter() - start) * 1e3, "bp_success", False, 0, 1.0, False, main_result)
        feature_pack = build_rescue_features(self.code, llr, main_result, snr_db, profile)
        if self.mode == "ai":
            policy = self._ai_policy(llr, feature_pack, snr_db, profile)
        else:
            policy = self._heuristic_policy(llr, feature_pack)
        gate = float(policy["gate"])
        rescue_p = float(policy.get("rescue", 0.0))
        if self.mode == "ai":
            hopeless_thr = float(self.cfg.get("hopeless_threshold", 0.08))
            rescue_thr = float(self.cfg.get("rescue_threshold", 0.30))
            gate_thr = float(self.cfg.get("gating_threshold", 0.35))
            expanded_thr = float(self.cfg.get("expanded_threshold", 0.55))
            should_rescue = (gate >= gate_thr) or (rescue_p >= rescue_thr) or (float(policy.get("expanded", 0.0)) >= expanded_thr)
            if (gate < hopeless_thr and rescue_p < rescue_thr) or (not should_rescue):
                return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), 0,
                                    (time.perf_counter() - start) * 1e3, "skip_hopeless", False, 0, gate, False, main_result)
        pool = _candidate_pool_from_feature_pack(feature_pack, int(policy["pool_size"]),
                                                 int(self.cfg["top_k_bits"]), int(self.cfg["top_k_unsat"]),
                                                 int(self.cfg["top_k_oscillation"]), bit_cost=np.asarray(policy["bit_cost"], dtype=np.float32))
        candidates = _enumerate_candidates(
            pool=pool,
            components=feature_pack["components"],
            bit_cost=np.asarray(policy["bit_cost"], dtype=np.float32),
            weight_candidates=policy["weight_candidates"],
            budget=int(policy["budget"]),
            weight_penalties=self.cfg["weight_penalties"],
            component_bonus=float(self.cfg.get("component_bonus", -0.25)),
        )
        valid: List[ValidCandidate] = []
        queries = 0
        used_micro = False
        failed_for_micro: List[tuple[np.ndarray, float, str, np.ndarray]] = []
        direct_budget = min(int(policy["budget"]), int(self.cfg.get("direct_budget", policy["budget"])))
        for mask_idx, prior_score, source in candidates:
            queries += 1
            mask = np.zeros(self.code.n, dtype=np.uint8)
            mask[mask_idx] = 1
            cand = main_result.hard.copy()
            cand[mask_idx] ^= 1
            syn = (self.code.h @ cand) % 2
            cand_feat = _mask_features(mask, feature_pack)
            if syn.sum() == 0:
                valid.append(ValidCandidate(mask, cand.copy(), float(prior_score), source, False, cand_feat))
            elif self.mode == "ai":
                failed_for_micro.append((mask.copy(), float(prior_score), source, cand_feat))
            if len(valid) >= int(self.cfg.get("rerank_list_size", 6)) and queries >= int(self.cfg.get("rerank_extra_queries", 16)):
                break
            if queries >= direct_budget:
                break

        if (not valid) and bool(self.cfg.get("enable_micro_bp", True)) and self.mode == "ai":
            micro_thr = float(self.cfg.get("micro_trigger_threshold", self.cfg.get("rescue_threshold", 0.30)))
            micro_cap = int(self.cfg.get("micro_candidate_cap", 2))
            if rescue_p >= micro_thr and failed_for_micro:
                shortlist = failed_for_micro
                if policy.get("candidate_net") is not None and policy.get("packet_emb") is not None:
                    cand_feats_np = np.stack([x[3] for x in failed_for_micro], axis=0).astype(np.float32)
                    cand_feats = torch.tensor(cand_feats_np[None, ...], dtype=torch.float32, device=self.device)
                    packet_emb = policy["packet_emb"].to(self.device)
                    with torch.no_grad():
                        micro_scores = policy["candidate_net"](packet_emb, cand_feats).cpu().numpy()[0]
                    order = np.argsort(-(micro_scores + float(self.cfg.get("rerank_prior_scale", 0.10)) * np.array([-x[1] for x in failed_for_micro], dtype=np.float32)))
                    shortlist = [failed_for_micro[i] for i in order[:micro_cap]]
                else:
                    shortlist = failed_for_micro[:micro_cap]
                for mask, prior_score, source, cand_feat in shortlist:
                    mask_idx = np.flatnonzero(mask)
                    llr_mod = np.asarray(llr, dtype=np.float32).copy()
                    llr_mod[mask_idx] *= -float(self.cfg.get("micro_flip_scale", 1.10))
                    micro = self.micro_bp.decode(llr_mod, max_iters=self.micro_bp.max_iters, collect_trace=False)
                    if micro.success:
                        used_micro = True
                        valid.append(ValidCandidate(mask, micro.hard.copy(), float(prior_score), source, True, cand_feat))
                        if len(valid) >= int(self.cfg.get("rerank_list_size", 6)):
                            break
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if not valid:
            return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), queries, elapsed_ms,
                                "rescue_fail", used_micro, 0, gate, True, main_result)
        if policy.get("candidate_net") is not None and policy.get("packet_emb") is not None:
            cand_feats = torch.tensor(np.stack([v.cand_features for v in valid], axis=0)[None, ...], dtype=torch.float32, device=self.device)
            packet_emb = policy["packet_emb"].to(self.device)
            with torch.no_grad():
                scores = policy["candidate_net"](packet_emb, cand_feats).cpu().numpy()[0]
            final_scores = scores + float(self.cfg.get("rerank_prior_scale", 0.12)) * np.array([-v.score for v in valid], dtype=np.float32)
            best_idx = int(np.argmax(final_scores))
        else:
            metrics = np.array([_channel_metric(llr, v.codeword) - 0.12 * v.score for v in valid], dtype=np.float32)
            best_idx = int(np.argmax(metrics))
        best = valid[best_idx]
        action = "rescue_success_micro" if best.used_micro_bp else "rescue_success_direct"
        return RescueResult(True, best.codeword.copy(), best.codeword.copy(), queries, elapsed_ms,
                            action, best.used_micro_bp, len(valid), gate, True, main_result)
