from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch

from ..codes.systematic_sparse import SystematicSparseCode
from ..models.posterior_scorer import PosteriorSchedulerNet, run_model_inference
from ..utils.combinatorics import score_combinations
from .baseline import WeightedReliabilityGRAND
from .features import build_decoder_features
from .results import DecodeAttempt, DecodeResult


@dataclass
class NeuroSymbolicGRAND:
    code: SystematicSparseCode
    model: PosteriorSchedulerNet
    device: str
    num_segments: int
    max_weight_class: int
    ai_pool_size: int
    ai_max_weight: int
    ai_budget: int
    fallback_decoder: WeightedReliabilityGRAND
    ai_weight_penalties: Sequence[float]
    top_segments: int
    top_bits_extra: int
    confidence_threshold: float
    candidate_mass_threshold: float
    adaptive_pool_size: int
    adaptive_max_weight: int
    adaptive_budget: int
    adaptive_top_segments: int
    adaptive_top_bits_extra: int
    overflow_expand_threshold: float
    overflow_direct_fallback_threshold: float
    confidence_presearch_threshold: float
    overflow_direct_action: str = "fallback"
    overflow_direct_confidence_ceiling: float = 1.0
    always_fallback_after_ai_fail: bool = True
    trace_top_attempts: int = 25

    def _choose_allowed_weights(self, weight_prob: np.ndarray, max_weight_cap: int) -> Set[int]:
        order = np.argsort(weight_prob)[::-1]
        chosen: Set[int] = set()
        mass = 0.0
        for idx in order:
            mass += float(weight_prob[idx])
            if idx <= max_weight_cap:
                chosen.add(int(idx))
            else:
                chosen.add(int(max_weight_cap))
            if mass >= self.candidate_mass_threshold:
                break
        if not chosen:
            chosen = {0, 1, min(2, max_weight_cap)}
        return chosen

    def _build_ai_pool(
        self,
        bit_prob_ranked: np.ndarray,
        rank_order: np.ndarray,
        top_segment_ids: np.ndarray,
        profile_segments: List[np.ndarray],
        pool_size: int,
        top_bits_extra: int,
    ) -> np.ndarray:
        active_ranks: Set[int] = set()
        for seg_id in top_segment_ids:
            active_ranks.update(profile_segments[int(seg_id)].tolist())
        top_prob_ranks = np.argsort(bit_prob_ranked)[::-1][: int(pool_size)]
        active_ranks.update(top_prob_ranks.tolist())
        active_ranks.update(np.arange(min(int(top_bits_extra), len(rank_order))).tolist())
        ranked_candidates = np.array(sorted(active_ranks, key=lambda idx: (-bit_prob_ranked[idx], idx)), dtype=np.int64)
        ranked_candidates = ranked_candidates[: int(pool_size)]
        return rank_order[ranked_candidates]

    @staticmethod
    def _shift_trace(trace: List[DecodeAttempt], offset: int) -> List[DecodeAttempt]:
        shifted: List[DecodeAttempt] = []
        for attempt in trace:
            shifted.append(
                DecodeAttempt(
                    stage=attempt.stage,
                    query_index=int(attempt.query_index) + int(offset),
                    weight=int(attempt.weight),
                    score=float(attempt.score),
                    positions=list(attempt.positions),
                    syndrome_weight=int(attempt.syndrome_weight),
                    success=bool(attempt.success),
                )
            )
        return shifted

    def _merge_fallback_result(
        self,
        *,
        fallback_result: DecodeResult,
        start_time: float,
        primary_queries: int,
        primary_elapsed_ms: float,
        candidate_pool_size: int,
        oracle_pool_hit: Optional[bool],
        predicted_weight_top: int,
        confidence_prob: float,
        predicted_overflow_prob: float,
        gate_reason: str,
        trace: List[DecodeAttempt],
        diagnostics: Dict[str, object],
    ) -> DecodeResult:
        total_elapsed_ms = (time.perf_counter() - start_time) * 1e3
        fallback_trace = self._shift_trace(fallback_result.trace[: self.trace_top_attempts], offset=primary_queries)
        merged_trace = (trace + fallback_trace)[: self.trace_top_attempts]
        return DecodeResult(
            success=bool(fallback_result.success),
            decoded_codeword=fallback_result.decoded_codeword.copy() if hasattr(fallback_result.decoded_codeword, "copy") else fallback_result.decoded_codeword,
            queries=int(primary_queries + fallback_result.queries),
            stage="fallback" if fallback_result.success else "fallback_fail",
            elapsed_ms=total_elapsed_ms,
            fallback_used=True,
            candidate_pool_size=int(candidate_pool_size),
            oracle_pool_hit=oracle_pool_hit,
            predicted_weight_top=int(predicted_weight_top),
            confidence_prob=float(confidence_prob),
            trace=merged_trace,
            primary_queries=int(primary_queries),
            fallback_queries=int(fallback_result.queries),
            primary_elapsed_ms=float(primary_elapsed_ms),
            fallback_elapsed_ms=float(fallback_result.elapsed_ms),
            gate_reason=str(gate_reason),
            predicted_overflow_prob=float(predicted_overflow_prob),
            diagnostics={
                **diagnostics,
                "fallback_success": bool(fallback_result.success),
                "fallback_stage": fallback_result.stage,
                "fallback_candidate_pool_size": int(fallback_result.candidate_pool_size),
                "fallback_budget": int(getattr(self.fallback_decoder, "budget", 0)),
                "fallback_pool_size": int(getattr(self.fallback_decoder, "pool_size", 0)),
                "fallback_max_weight": int(getattr(self.fallback_decoder, "max_weight", 0)),
            },
        )

    def _skip_result(
        self,
        *,
        hard_bits: np.ndarray,
        start_time: float,
        predicted_weight_top: int,
        confidence_prob: float,
        predicted_overflow_prob: float,
        gate_reason: str,
        diagnostics: Dict[str, object],
    ) -> DecodeResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1e3
        return DecodeResult(
            success=False,
            decoded_codeword=hard_bits.copy(),
            queries=0,
            stage="skip_hopeless",
            elapsed_ms=elapsed_ms,
            fallback_used=False,
            candidate_pool_size=0,
            oracle_pool_hit=None,
            predicted_weight_top=int(predicted_weight_top),
            confidence_prob=float(confidence_prob),
            trace=[],
            primary_queries=0,
            fallback_queries=0,
            primary_elapsed_ms=elapsed_ms,
            fallback_elapsed_ms=0.0,
            gate_reason=str(gate_reason),
            predicted_overflow_prob=float(predicted_overflow_prob),
            diagnostics=diagnostics,
        )

    def decode(
        self,
        llr: np.ndarray,
        hard_bits: np.ndarray,
        snr_db: float,
        profile_id: int,
        truth_error_mask: Optional[np.ndarray] = None,
    ) -> DecodeResult:
        start = time.perf_counter()
        llr = np.asarray(llr, dtype=np.float32)
        hard_bits = np.asarray(hard_bits, dtype=np.uint8)
        if self.code.is_codeword(hard_bits):
            elapsed = (time.perf_counter() - start) * 1e3
            return DecodeResult(
                success=True,
                decoded_codeword=hard_bits.copy(),
                queries=0,
                stage="hard_decision",
                elapsed_ms=elapsed,
                fallback_used=False,
                candidate_pool_size=0,
                confidence_prob=1.0,
                primary_queries=0,
                fallback_queries=0,
                primary_elapsed_ms=elapsed,
                fallback_elapsed_ms=0.0,
                gate_reason="hard_decision",
                predicted_overflow_prob=0.0,
                diagnostics={
                    "search_mode": "hard_decision",
                    "policy_action": "hard_decision",
                    "allowed_weights": [0],
                    "top_segment_ids": [],
                    "pool_positions_top10": [],
                    "primary_budget": 0,
                    "primary_max_weight": 0,
                    "presearch_gate": False,
                },
            )

        feat = build_decoder_features(
            llr=llr,
            hard_bits=hard_bits,
            code=self.code,
            snr_db=snr_db,
            profile_id=profile_id,
            num_segments=self.num_segments,
        )
        bit_tensor = torch.from_numpy(feat.bit_features).unsqueeze(0).to(self.device)
        global_tensor = torch.from_numpy(feat.global_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            infer = run_model_inference(self.model, bit_tensor, global_tensor)
        bit_prob_ranked = infer.bit_flip_prob.squeeze(0).cpu().numpy().astype(np.float64)
        segment_prob = infer.segment_prob.squeeze(0).cpu().numpy().astype(np.float64)
        weight_prob = infer.weight_prob.squeeze(0).cpu().numpy().astype(np.float64)
        confidence_prob = float(infer.confidence_prob.squeeze(0).cpu().item())
        predicted_overflow_prob = float(weight_prob[-1])
        predicted_weight_top = int(np.argmax(weight_prob))

        bounds = np.linspace(0, self.code.n, self.num_segments + 1, dtype=int)
        profile_segments = [np.arange(bounds[i], bounds[i + 1], dtype=np.int64) for i in range(self.num_segments)]

        direct_threshold = float(self.overflow_direct_fallback_threshold)
        direct_action = str(self.overflow_direct_action).strip().lower()
        direct_conf_ceiling = float(self.overflow_direct_confidence_ceiling)
        if predicted_overflow_prob >= direct_threshold and confidence_prob <= direct_conf_ceiling:
            if direct_action == "skip":
                return self._skip_result(
                    hard_bits=hard_bits,
                    start_time=start,
                    predicted_weight_top=predicted_weight_top,
                    confidence_prob=confidence_prob,
                    predicted_overflow_prob=predicted_overflow_prob,
                    gate_reason="presearch_skip_hopeless",
                    diagnostics={
                        "search_mode": "skip_hopeless",
                        "policy_action": "presearch_skip_hopeless",
                        "allowed_weights": [],
                        "top_segment_ids": [],
                        "pool_positions_top10": [],
                        "primary_budget": 0,
                        "primary_max_weight": 0,
                        "presearch_gate": True,
                        "predicted_overflow_prob": predicted_overflow_prob,
                    },
                )
            if direct_action == "fallback":
                fallback_result = self.fallback_decoder.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
                return self._merge_fallback_result(
                    fallback_result=fallback_result,
                    start_time=start,
                    primary_queries=0,
                    primary_elapsed_ms=0.0,
                    candidate_pool_size=0,
                    oracle_pool_hit=None,
                    predicted_weight_top=predicted_weight_top,
                    confidence_prob=confidence_prob,
                    predicted_overflow_prob=predicted_overflow_prob,
                    gate_reason="presearch_overflow_direct",
                    trace=[],
                    diagnostics={
                        "search_mode": "direct_fallback",
                        "policy_action": "presearch_direct_fallback",
                        "allowed_weights": [],
                        "top_segment_ids": [],
                        "pool_positions_top10": [],
                        "primary_budget": 0,
                        "primary_max_weight": 0,
                        "presearch_gate": True,
                        "predicted_overflow_prob": predicted_overflow_prob,
                    },
                )

        if confidence_prob < float(self.confidence_presearch_threshold):
            fallback_result = self.fallback_decoder.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
            return self._merge_fallback_result(
                fallback_result=fallback_result,
                start_time=start,
                primary_queries=0,
                primary_elapsed_ms=0.0,
                candidate_pool_size=0,
                oracle_pool_hit=None,
                predicted_weight_top=predicted_weight_top,
                confidence_prob=confidence_prob,
                predicted_overflow_prob=predicted_overflow_prob,
                gate_reason="presearch_low_confidence",
                trace=[],
                diagnostics={
                    "search_mode": "direct_fallback",
                    "policy_action": "presearch_low_confidence_fallback",
                    "allowed_weights": [],
                    "top_segment_ids": [],
                    "pool_positions_top10": [],
                    "primary_budget": 0,
                    "primary_max_weight": 0,
                    "presearch_gate": True,
                    "predicted_overflow_prob": predicted_overflow_prob,
                },
            )

        search_mode = "standard"
        policy_action = "ai_search"
        pool_size = int(self.ai_pool_size)
        max_weight = int(self.ai_max_weight)
        budget = int(self.ai_budget)
        top_segments = int(self.top_segments)
        top_bits_extra = int(self.top_bits_extra)

        if predicted_overflow_prob >= float(self.overflow_expand_threshold):
            search_mode = "expanded_overflow"
            policy_action = "expanded_ai_search"
            pool_size = int(self.adaptive_pool_size)
            max_weight = int(self.adaptive_max_weight)
            budget = int(self.adaptive_budget)
            top_segments = int(self.adaptive_top_segments)
            top_bits_extra = int(self.adaptive_top_bits_extra)

        top_segment_ids = np.argsort(segment_prob)[::-1][:top_segments]
        pool_positions = self._build_ai_pool(
            bit_prob_ranked,
            feat.rank_order,
            top_segment_ids,
            profile_segments,
            pool_size=pool_size,
            top_bits_extra=top_bits_extra,
        )
        pool_ranks = feat.inverse_rank_order[pool_positions]
        allowed_weights = self._choose_allowed_weights(weight_prob, max_weight_cap=max_weight)
        candidate_scores = -np.log(np.clip(bit_prob_ranked[pool_ranks], 1e-6, 1.0 - 1e-6))
        scored = score_combinations(
            pool_scores=candidate_scores,
            max_weight=max_weight,
            allowed_weights=allowed_weights,
            weight_penalties=self.ai_weight_penalties,
            top_k=budget,
        )

        oracle_pool_hit = None
        if truth_error_mask is not None:
            truth_positions = set(np.where(np.asarray(truth_error_mask, dtype=np.uint8) == 1)[0].tolist())
            oracle_pool_hit = truth_positions.issubset(set(pool_positions.tolist()))

        trace: List[DecodeAttempt] = []
        queries = 0
        diagnostics = {
            "search_mode": search_mode,
            "policy_action": policy_action,
            "allowed_weights": sorted(int(x) for x in allowed_weights),
            "top_segment_ids": top_segment_ids.astype(int).tolist(),
            "pool_positions_top10": pool_positions[:10].astype(int).tolist(),
            "primary_budget": int(budget),
            "primary_max_weight": int(max_weight),
            "presearch_gate": False,
            "predicted_overflow_prob": predicted_overflow_prob,
            "predicted_weight_distribution": weight_prob.tolist(),
            "top_ranked_bit_prob": np.argsort(bit_prob_ranked)[::-1][:10].astype(int).tolist(),
        }
        primary_start = time.perf_counter()
        for score, combo in scored:
            if len(combo) == 0:
                continue
            mask = np.zeros(self.code.n, dtype=np.uint8)
            positions = pool_positions[list(combo)]
            mask[positions] = 1
            candidate = hard_bits ^ mask
            syndrome_weight = int(self.code.syndrome(candidate)[0].sum())
            queries += 1
            success = syndrome_weight == 0
            if len(trace) < self.trace_top_attempts:
                trace.append(
                    DecodeAttempt(
                        stage="ai",
                        query_index=queries,
                        weight=len(combo),
                        score=float(score),
                        positions=positions.astype(int).tolist(),
                        syndrome_weight=syndrome_weight,
                        success=success,
                    )
                )
            if success:
                primary_elapsed_ms = (time.perf_counter() - primary_start) * 1e3
                total_elapsed_ms = (time.perf_counter() - start) * 1e3
                return DecodeResult(
                    success=True,
                    decoded_codeword=candidate,
                    queries=queries,
                    stage="ai",
                    elapsed_ms=total_elapsed_ms,
                    fallback_used=False,
                    candidate_pool_size=len(pool_positions),
                    oracle_pool_hit=oracle_pool_hit,
                    predicted_weight_top=predicted_weight_top,
                    confidence_prob=confidence_prob,
                    trace=trace,
                    primary_queries=queries,
                    fallback_queries=0,
                    primary_elapsed_ms=primary_elapsed_ms,
                    fallback_elapsed_ms=0.0,
                    gate_reason="ai_success",
                    predicted_overflow_prob=predicted_overflow_prob,
                    diagnostics={**diagnostics, "ai_success": True, "fallback_success": False},
                )

        primary_elapsed_ms = (time.perf_counter() - primary_start) * 1e3
        if self.always_fallback_after_ai_fail or confidence_prob < float(self.confidence_threshold):
            fallback_result = self.fallback_decoder.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
            return self._merge_fallback_result(
                fallback_result=fallback_result,
                start_time=start,
                primary_queries=queries,
                primary_elapsed_ms=primary_elapsed_ms,
                candidate_pool_size=len(pool_positions),
                oracle_pool_hit=oracle_pool_hit,
                predicted_weight_top=predicted_weight_top,
                confidence_prob=confidence_prob,
                predicted_overflow_prob=predicted_overflow_prob,
                gate_reason="postsearch_exhausted",
                trace=trace,
                diagnostics={**diagnostics, "policy_action": "postsearch_fallback", "ai_success": False, "fallback_success": bool(fallback_result.success)},
            )

        total_elapsed_ms = (time.perf_counter() - start) * 1e3
        return DecodeResult(
            success=False,
            decoded_codeword=hard_bits.copy(),
            queries=queries,
            stage="ai_fail",
            elapsed_ms=total_elapsed_ms,
            fallback_used=False,
            candidate_pool_size=len(pool_positions),
            oracle_pool_hit=oracle_pool_hit,
            predicted_weight_top=predicted_weight_top,
            confidence_prob=confidence_prob,
            trace=trace,
            primary_queries=queries,
            fallback_queries=0,
            primary_elapsed_ms=primary_elapsed_ms,
            fallback_elapsed_ms=0.0,
            gate_reason="ai_exhausted_no_fallback",
            predicted_overflow_prob=predicted_overflow_prob,
            diagnostics={**diagnostics, "policy_action": "ai_fail_no_fallback", "ai_success": False, "fallback_success": False},
        )
