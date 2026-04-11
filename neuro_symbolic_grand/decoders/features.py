from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codes.systematic_sparse import SystematicSparseCode


@dataclass
class DecoderFeatures:
    rank_order: np.ndarray
    inverse_rank_order: np.ndarray
    bit_features: np.ndarray
    global_features: np.ndarray
    hard_ranked: np.ndarray
    llr_ranked: np.ndarray
    error_mask_ranked: np.ndarray | None = None
    segment_labels: np.ndarray | None = None
    weight_label: int | None = None
    confidence_label: float | None = None


def _segment_labels(error_mask_ranked: np.ndarray, num_segments: int) -> np.ndarray:
    n = len(error_mask_ranked)
    bounds = np.linspace(0, n, num_segments + 1, dtype=int)
    labels = np.zeros(num_segments, dtype=np.float32)
    for idx in range(num_segments):
        start, stop = bounds[idx], bounds[idx + 1]
        labels[idx] = float(error_mask_ranked[start:stop].sum() > 0)
    return labels


def build_decoder_features(
    llr: np.ndarray,
    hard_bits: np.ndarray,
    code: SystematicSparseCode,
    snr_db: float,
    profile_id: int,
    num_segments: int,
    error_mask: np.ndarray | None = None,
    confidence_weight_threshold: int = 4,
    confidence_rank_limit: int | None = None,
) -> DecoderFeatures:
    llr = np.asarray(llr, dtype=np.float32)
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    rank_order = np.argsort(np.abs(llr), kind="stable")
    inverse_rank_order = np.empty_like(rank_order)
    inverse_rank_order[rank_order] = np.arange(len(rank_order), dtype=np.int64)

    llr_ranked = llr[rank_order]
    hard_ranked = hard_bits[rank_order]
    unsat_counts = code.unsatisfied_check_counts(hard_bits)[rank_order].astype(np.float32)
    column_weights = code.column_weights[rank_order].astype(np.float32)
    syndrome = code.syndrome(hard_bits)[0].astype(np.float32)

    max_abs_llr = float(np.max(np.abs(llr_ranked))) if llr_ranked.size else 1.0
    max_unsat = float(max(1.0, unsat_counts.max(initial=0.0)))
    max_cw = float(max(1.0, column_weights.max(initial=0.0)))
    rank_fraction = np.linspace(0.0, 1.0, len(rank_order), dtype=np.float32)

    bit_features = np.stack(
        [
            np.abs(llr_ranked) / max(max_abs_llr, 1e-6),
            np.sign(llr_ranked),
            hard_ranked.astype(np.float32),
            unsat_counts / max_unsat,
            column_weights / max_cw,
            rank_fraction,
        ],
        axis=1,
    ).astype(np.float32)

    global_features = np.array(
        [
            float(snr_db) / 12.0,
            float(profile_id) / 8.0,
            float(syndrome.sum()) / max(1.0, len(syndrome)),
            float(code.rate),
        ],
        dtype=np.float32,
    )

    error_mask_ranked = None
    segment_labels = None
    weight_label = None
    confidence_label = None
    if error_mask is not None:
        error_mask_ranked = np.asarray(error_mask, dtype=np.uint8)[rank_order]
        segment_labels = _segment_labels(error_mask_ranked, num_segments=num_segments)
        weight = int(error_mask_ranked.sum())
        weight_label = weight
        if weight == 0:
            confidence_label = 1.0
        else:
            in_weight = weight <= int(confidence_weight_threshold)
            if confidence_rank_limit is None:
                in_rank = True
            else:
                error_ranks = np.where(error_mask_ranked == 1)[0]
                in_rank = bool(error_ranks.size > 0 and error_ranks.max() < int(confidence_rank_limit))
            confidence_label = float(in_weight and in_rank)

    return DecoderFeatures(
        rank_order=rank_order.astype(np.int64),
        inverse_rank_order=inverse_rank_order.astype(np.int64),
        bit_features=bit_features,
        global_features=global_features,
        hard_ranked=hard_ranked.astype(np.uint8),
        llr_ranked=llr_ranked.astype(np.float32),
        error_mask_ranked=error_mask_ranked,
        segment_labels=segment_labels,
        weight_label=weight_label,
        confidence_label=confidence_label,
    )
