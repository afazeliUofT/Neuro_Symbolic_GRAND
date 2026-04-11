from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np
import torch

from ..codes.systematic_sparse import SystematicSparseCode
from ..models.posterior_scorer import PosteriorSchedulerNet, run_model_inference
from ..utils.combinatorics import score_combinations
from ..utils.math_utils import sigmoid
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
    trace_top_attempts: int = 25

    def _choose_allowed_weights(self, weight_prob: np.ndarray) -> Set[int]:
        order = np.argsort(weight_prob)[::-1]
        chosen: Set[int] = set()
        mass = 0.0
        for idx in order:
            mass += float(weight_prob[idx])
            if idx <= self.ai_max_weight:
                chosen.add(int(idx))
            else:
                chosen.add(self.ai_max_weight)
            if mass >= self.candidate_mass_threshold:
                break
        if not chosen:
            chosen = {0, 1, 2}
        return chosen

    def _build_ai_pool(
        self,
        bit_prob_ranked: np.ndarray,
        rank_order: np.ndarray,
        top_segment_ids: np.ndarray,
        profile_segments: List[np.ndarray],
    ) -> np.ndarray:
        active_ranks: Set[int] = set()
        for seg_id in top_segment_ids:
            active_ranks.update(profile_segments[int(seg_id)].tolist())
        top_prob_ranks = np.argsort(bit_prob_ranked)[::-1][: self.ai_pool_size]
        active_ranks.update(top_prob_ranks.tolist())
        active_ranks.update(np.arange(min(self.top_bits_extra, len(rank_order))).tolist())
        ranked_candidates = np.array(sorted(active_ranks, key=lambda idx: (-bit_prob_ranked[idx], idx)), dtype=np.int64)
        ranked_candidates = ranked_candidates[: self.ai_pool_size]
        return rank_order[ranked_candidates]

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
            return DecodeResult(
                success=True,
                decoded_codeword=hard_bits.copy(),
                queries=0,
                stage="hard_decision",
                elapsed_ms=(time.perf_counter() - start) * 1e3,
                fallback_used=False,
                candidate_pool_size=0,
                confidence_prob=1.0,
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

        bounds = np.linspace(0, self.code.n, self.num_segments + 1, dtype=int)
        profile_segments = [np.arange(bounds[i], bounds[i + 1], dtype=np.int64) for i in range(self.num_segments)]
        top_segment_ids = np.argsort(segment_prob)[::-1][: self.top_segments]
        pool_positions = self._build_ai_pool(bit_prob_ranked, feat.rank_order, top_segment_ids, profile_segments)
        pool_ranks = feat.inverse_rank_order[pool_positions]
        allowed_weights = self._choose_allowed_weights(weight_prob)
        candidate_scores = -np.log(np.clip(bit_prob_ranked[pool_ranks], 1e-6, 1.0 - 1e-6))
        scored = score_combinations(
            pool_scores=candidate_scores,
            max_weight=self.ai_max_weight,
            allowed_weights=allowed_weights,
            weight_penalties=self.ai_weight_penalties,
            top_k=self.ai_budget,
        )

        oracle_pool_hit = None
        if truth_error_mask is not None:
            truth_positions = set(np.where(np.asarray(truth_error_mask, dtype=np.uint8) == 1)[0].tolist())
            oracle_pool_hit = truth_positions.issubset(set(pool_positions.tolist()))

        trace: List[DecodeAttempt] = []
        queries = 0
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
                return DecodeResult(
                    success=True,
                    decoded_codeword=candidate,
                    queries=queries,
                    stage="ai",
                    elapsed_ms=(time.perf_counter() - start) * 1e3,
                    fallback_used=False,
                    candidate_pool_size=len(pool_positions),
                    oracle_pool_hit=oracle_pool_hit,
                    predicted_weight_top=int(np.argmax(weight_prob)),
                    confidence_prob=confidence_prob,
                    trace=trace,
                )

        if confidence_prob < self.confidence_threshold:
            fallback_result = self.fallback_decoder.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=truth_error_mask)
            fallback_result.elapsed_ms = (time.perf_counter() - start) * 1e3
            fallback_result.fallback_used = True
            fallback_result.candidate_pool_size = len(pool_positions)
            fallback_result.oracle_pool_hit = oracle_pool_hit
            fallback_result.predicted_weight_top = int(np.argmax(weight_prob))
            fallback_result.confidence_prob = confidence_prob
            fallback_result.trace = trace + fallback_result.trace[: self.trace_top_attempts]
            return fallback_result

        return DecodeResult(
            success=False,
            decoded_codeword=hard_bits.copy(),
            queries=queries,
            stage="ai_fail",
            elapsed_ms=(time.perf_counter() - start) * 1e3,
            fallback_used=False,
            candidate_pool_size=len(pool_positions),
            oracle_pool_hit=oracle_pool_hit,
            predicted_weight_top=int(np.argmax(weight_prob)),
            confidence_prob=confidence_prob,
            trace=trace,
        )
