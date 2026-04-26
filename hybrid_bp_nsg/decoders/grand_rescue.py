from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from ..codes.peg_ldpc import LDPCCode
from ..decoders.bp import BeliefPropagationDecoder, BPDecodeResult
from ..rescue_search import (
    candidate_feature_vector,
    candidate_pool_from_feature_pack,
    channel_metric_internal,
    enumerate_mask_candidates,
    greedy_syndrome_repair_candidates,
    syndrome_osd_candidates,
)
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
    crc_ok: bool = True
    selected_valid_codeword: bool = False
    crc_valid_candidates: int = 0
    parity_valid_candidates: int = 0


@dataclass
class ValidCandidate:
    mask: np.ndarray
    codeword: np.ndarray
    score: float
    source: str
    used_micro_bp: bool
    cand_features: np.ndarray
    crc_ok: bool
    channel_metric: float


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
        channel_unreliability_bonus = -0.25 * (1.0 - np.clip(vf[:, 3], 0.0, 1.0))
        syndrome_bonus = -0.18 * np.clip(vf[:, 11], 0.0, 1.0)
        disagreement_bonus = -0.10 * vf[:, 13]
        puncture_bonus = -0.05 * vf[:, 14]
        bit_cost = np.log1p(inv_cost * 10.0) + 0.03 * abs_llr / max(1e-6, float(abs_llr.max()))
        bit_cost = bit_cost + channel_unreliability_bonus + syndrome_bonus + disagreement_bonus + puncture_bonus

        if self.mode in {"orb", "heuristic"}:
            weight_candidates = list(range(1, min(6, int(self.cfg["max_standard_weight"])) + 1))
            pool_size = min(int(self.cfg["pool_size"]), max(16, int(self.cfg.get("top_k_bits", 64))))
            budget = min(int(self.cfg["standard_budget"]), int(self.cfg.get("direct_budget", self.cfg["standard_budget"])))
        else:
            max_exp = int(self.cfg["max_expanded_weight"])
            weight_candidates = list(range(1, min(max_exp, 10) + 1))
            pool_size = int(self.cfg["expanded_pool_size"])
            budget = int(self.cfg["expanded_budget"])

        return {
            "bit_cost": bit_cost.astype(np.float32),
            "weight_candidates": weight_candidates,
            "pool_size": pool_size,
            "budget": budget,
            "gate": 1.0,
            "expanded": 1.0,
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
                "candidate_features": tf.zeros((1, 1, 14), dtype=tf.float32),
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
        rank_cost = np.linspace(0.0, 1.0, self.code.n, endpoint=False, dtype=np.float32)
        inv_cost = np.empty_like(rank_cost)
        inv_cost[feature_pack["heuristic_order"]] = rank_cost
        bit_cost += 0.08 * inv_cost
        bit_cost -= float(self.cfg.get("segment_bonus_scale", 0.0)) * seg_prob[seg_idx]
        bit_cost -= float(self.cfg.get("component_focus_scale", 0.0)) * vf[:, -1]
        bit_cost -= float(self.cfg.get("oscillation_focus_scale", 0.0)) * vf[:, 7]
        bit_cost -= 0.10 * vf[:, 11]
        bit_cost -= 0.06 * vf[:, 13]
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
        pool = candidate_pool_from_feature_pack(
            feature_pack,
            int(policy["pool_size"]),
            int(self.cfg["top_k_bits"]),
            int(self.cfg["top_k_unsat"]),
            int(self.cfg["top_k_oscillation"]),
            bit_cost=bit_cost,
        )
        candidates = enumerate_mask_candidates(
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
            candidates.extend(greedy_syndrome_repair_candidates(
                self.code, base_hard, bit_cost,
                max_steps=int(self.cfg.get("greedy_repair_steps", 32)),
                max_candidates=int(self.cfg.get("greedy_repair_candidates", 6)),
            ))
        if bool(self.cfg.get("enable_osd_repair", True)):
            candidates.extend(syndrome_osd_candidates(
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

        def add_candidate(mask: np.ndarray, cand: np.ndarray, prior_score: float, source: str, used_micro_bp: bool) -> None:
            crc_ok = self.code.crc_check_internal(cand)
            feat = candidate_feature_vector(self.code, llr, base_hard, cand, mask, feature_pack, prior_score, crc_ok)
            valid.append(ValidCandidate(mask.copy(), cand.copy(), float(prior_score), source, used_micro_bp, feat, crc_ok, channel_metric_internal(llr, cand)))

        if int(self.code.syndrome(base_hard).sum()) == 0:
            add_candidate(np.zeros(self.code.n, dtype=np.uint8), base_hard.copy(), 0.0, "base_codeword", False)

        seen = set()
        base_syn = self.code.syndrome(base_hard).astype(np.uint8)
        h_int = self.code.h.astype(np.int16, copy=False)
        parallel_bs = max(1, int(self.cfg.get("parallel_test_batch_size", 256)))
        cand_iter = []
        for mask_idx, prior_score, source in candidates:
            key = tuple(sorted(int(x) for x in mask_idx))
            if key in seen:
                continue
            seen.add(key)
            cand_iter.append((key, float(prior_score), source))
            if len(cand_iter) >= direct_budget:
                break

        for start_idx in range(0, len(cand_iter), parallel_bs):
            chunk = cand_iter[start_idx : start_idx + parallel_bs]
            if not chunk:
                continue
            bsz = len(chunk)
            dense_masks = np.zeros((bsz, self.code.n), dtype=np.uint8)
            for bi, (key, _prior_score, _source) in enumerate(chunk):
                dense_masks[bi, list(key)] = 1
            # Parallel parity test: syn(base xor mask) = syn(base) xor H*mask. A candidate
            # is parity-valid iff H*mask == syn(base). Using a dense batch here is much
            # faster than testing masks one-by-one and lets BLAS/threaded matmul use all CPUs.
            delta = (h_int @ dense_masks.T.astype(np.int16)) % 2
            valid_flags = np.all(delta == base_syn[:, None], axis=0)
            cand_batch = (base_hard[None, :] ^ dense_masks).astype(np.uint8)
            queries += bsz
            for bi, is_valid in enumerate(valid_flags.tolist()):
                key, prior_score, source = chunk[bi]
                mask = dense_masks[bi]
                cand = cand_batch[bi]
                if is_valid:
                    add_candidate(mask, cand, prior_score, source, False)
                elif self.mode == "ai":
                    feat = candidate_feature_vector(self.code, llr, base_hard, cand, mask, feature_pack, prior_score, False)
                    failed_for_micro.append((mask.copy(), float(prior_score), source, feat))
            if len(valid) >= int(self.cfg.get("rerank_list_size", 12)) and queries >= int(self.cfg.get("rerank_extra_queries", 32)):
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
                for mask, prior_score, source, _ in shortlist:
                    mask_idx = np.flatnonzero(mask)
                    llr_mod = np.asarray(llr, dtype=np.float32).copy()
                    llr_mod[mask_idx] *= -float(self.cfg.get("micro_flip_scale", 1.10))
                    micro = self.micro_bp.decode(llr_mod, max_iters=self.micro_bp.max_iters, collect_trace=False)
                    if int(self.code.syndrome(micro.hard).sum()) == 0:
                        add_candidate(mask, micro.hard.copy(), float(prior_score), source + "_micro", True)
                        used_micro = True
                    if len(valid) >= int(self.cfg.get("rerank_list_size", 12)):
                        break
        return valid, queries, used_micro

    def rescue(self, llr: np.ndarray, main_result: BPDecodeResult, snr_db: float, profile: str) -> RescueResult:
        start = time.perf_counter()
        if main_result.success:
            return RescueResult(True, main_result.hard.copy(), main_result.hard.copy(), 0,
                                (time.perf_counter() - start) * 1e3, "bp_success", False, 0, 1.0, False, main_result,
                                crc_ok=bool(main_result.crc_ok), selected_valid_codeword=True, crc_valid_candidates=1, parity_valid_candidates=1)

        target_basis = str(self.cfg.get("target_basis", "channel_with_bp_punctures"))
        feature_pack = build_rescue_features(
            self.code, llr, main_result, snr_db, profile,
            num_segments=int(self.cfg.get("num_segments", 8)),
            target_basis=target_basis,
        )
        policy = self._ai_policy(llr, feature_pack, snr_db, profile) if self.mode == "ai" else self._heuristic_policy(llr, feature_pack)

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
                                    (time.perf_counter() - start) * 1e3, "skip_hopeless", False, 0, gate, False, main_result,
                                    crc_ok=False, selected_valid_codeword=False)

        base_hard = np.asarray(feature_pack["grand_base_hard"], dtype=np.uint8)
        valid, queries, used_micro = self._run_search_on_base(llr, base_hard, main_result, feature_pack, policy, rescue_p)

        if (not valid) and bool(self.cfg.get("try_bp_basis_fallback", True)):
            fp2 = build_rescue_features(
                self.code, llr, main_result, snr_db, profile,
                num_segments=int(self.cfg.get("num_segments", 8)),
                target_basis="bp",
            )
            valid2, q2, micro2 = self._run_search_on_base(llr, np.asarray(main_result.hard, dtype=np.uint8), main_result, fp2, policy, rescue_p)
            valid.extend(valid2)
            queries += q2
            used_micro = used_micro or micro2

        elapsed_ms = (time.perf_counter() - start) * 1e3
        if not valid:
            return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), queries, elapsed_ms,
                                "rescue_fail", used_micro, 0, gate, True, main_result,
                                crc_ok=False, selected_valid_codeword=False, crc_valid_candidates=0, parity_valid_candidates=0)

        parity_valid_candidates = len(valid)
        crc_valid = [v for v in valid if (v.crc_ok or not self.code.has_outer_crc)]
        crc_valid_candidates = len(crc_valid)
        require_crc = bool(self.cfg.get("require_crc_for_accept", True)) and self.code.has_outer_crc
        candidate_set = crc_valid if (require_crc and crc_valid) else valid

        if policy.get("candidate_net") is not None and policy.get("packet_emb") is not None and tf is not None:
            with tf.device(self.device):
                cand_feats = tf.convert_to_tensor(np.stack([v.cand_features for v in candidate_set], axis=0)[None, ...], dtype=tf.float32)
                packet_emb = policy["packet_emb"]
                scores = policy["candidate_net"](packet_emb, cand_feats, training=False).numpy()[0]
            final_scores = (
                scores
                + float(self.cfg.get("rerank_prior_scale", 0.10)) * np.array([-v.score for v in candidate_set], dtype=np.float32)
                + np.array([v.channel_metric for v in candidate_set], dtype=np.float32) / max(1, self.code.n)
                + np.array([1.5 if v.crc_ok else -1.5 for v in candidate_set], dtype=np.float32)
            )
            best_idx = int(np.argmax(final_scores))
        else:
            metrics = np.array([
                v.channel_metric - 0.10 * v.score + (1.5 if v.crc_ok else -1.5) for v in candidate_set
            ], dtype=np.float32)
            best_idx = int(np.argmax(metrics))

        best = candidate_set[best_idx]
        if require_crc and not best.crc_ok:
            return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), queries, elapsed_ms,
                                "reject_crc_fail", used_micro, len(valid), gate, True, main_result,
                                crc_ok=False, selected_valid_codeword=False,
                                crc_valid_candidates=crc_valid_candidates, parity_valid_candidates=parity_valid_candidates)

        if require_crc and crc_valid_candidates == 0:
            return RescueResult(False, main_result.hard.copy(), main_result.hard.copy(), queries, elapsed_ms,
                                "no_crc_valid_candidate", used_micro, len(valid), gate, True, main_result,
                                crc_ok=False, selected_valid_codeword=False,
                                crc_valid_candidates=crc_valid_candidates, parity_valid_candidates=parity_valid_candidates)

        action = "rescue_success_micro" if best.used_micro_bp else f"rescue_success_{best.source}"
        return RescueResult(True, best.codeword.copy(), best.codeword.copy(), queries, elapsed_ms,
                            action, best.used_micro_bp, len(valid), gate, True, main_result,
                            crc_ok=best.crc_ok, selected_valid_codeword=True,
                            crc_valid_candidates=crc_valid_candidates, parity_valid_candidates=parity_valid_candidates)
