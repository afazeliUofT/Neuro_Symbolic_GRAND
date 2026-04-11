from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..codes.systematic_sparse import SystematicSparseCode
from ..utils.combinatorics import score_combinations
from .results import DecodeAttempt, DecodeResult


@dataclass
class WeightedReliabilityGRAND:
    code: SystematicSparseCode
    pool_size: int
    max_weight: int
    budget: int
    weight_penalties: List[float]
    trace_top_attempts: int = 25

    def decode(
        self,
        llr: np.ndarray,
        hard_bits: np.ndarray,
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
                oracle_pool_hit=True if truth_error_mask is not None and int(np.sum(truth_error_mask)) == 0 else None,
                trace=[],
            )

        rank_order = np.argsort(np.abs(llr), kind="stable")
        pool_positions = rank_order[: min(self.pool_size, len(rank_order))]
        pool_scores = np.abs(llr[pool_positions]).astype(np.float64)
        scored = score_combinations(
            pool_scores=pool_scores,
            max_weight=self.max_weight,
            weight_penalties=self.weight_penalties,
            top_k=self.budget,
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
                        stage="baseline",
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
                    stage="baseline",
                    elapsed_ms=(time.perf_counter() - start) * 1e3,
                    fallback_used=False,
                    candidate_pool_size=len(pool_positions),
                    oracle_pool_hit=oracle_pool_hit,
                    trace=trace,
                )

        return DecodeResult(
            success=False,
            decoded_codeword=hard_bits.copy(),
            queries=queries,
            stage="baseline_fail",
            elapsed_ms=(time.perf_counter() - start) * 1e3,
            fallback_used=False,
            candidate_pool_size=len(pool_positions),
            oracle_pool_hit=oracle_pool_hit,
            trace=trace,
        )
