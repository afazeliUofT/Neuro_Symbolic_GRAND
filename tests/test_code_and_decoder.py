from __future__ import annotations

import numpy as np
import torch

from neuro_symbolic_grand.codes import build_systematic_sparse_code
from neuro_symbolic_grand.decoders import NeuroSymbolicGRAND, WeightedReliabilityGRAND
from neuro_symbolic_grand.decoders.features import build_decoder_features
from neuro_symbolic_grand.models.posterior_scorer import PosteriorSchedulerNet


class DummyModel(torch.nn.Module):
    def __init__(self, n_bits: int, num_segments: int, max_weight_class: int, *, confidence: float, overflow_prob: float, hot_rank: int) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.num_segments = num_segments
        self.max_weight_class = max_weight_class
        self.confidence = confidence
        self.overflow_prob = overflow_prob
        self.hot_rank = hot_rank

    def forward(self, bit_features: torch.Tensor, global_features: torch.Tensor):
        batch_size, n_bits, _ = bit_features.shape
        bit_logits = torch.full((batch_size, n_bits), -6.0, dtype=bit_features.dtype, device=bit_features.device)
        bit_logits[:, self.hot_rank] = 6.0
        segment_logits = torch.zeros((batch_size, self.num_segments), dtype=bit_features.dtype, device=bit_features.device)
        weight_logits = torch.full((batch_size, self.max_weight_class + 2), -6.0, dtype=bit_features.dtype, device=bit_features.device)
        if self.overflow_prob > 0.5:
            weight_logits[:, -1] = 6.0
        else:
            weight_logits[:, 1] = 6.0
        confidence_logit = torch.full((batch_size,), np.log(self.confidence / max(1e-6, 1.0 - self.confidence)), dtype=bit_features.dtype, device=bit_features.device)
        return {
            "bit_logits": bit_logits,
            "segment_logits": segment_logits,
            "weight_logits": weight_logits,
            "confidence_logit": confidence_logit,
        }


def test_systematic_code_encode_has_zero_syndrome() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=1)
    messages = np.random.default_rng(0).integers(0, 2, size=(10, code.k), dtype=np.uint8)
    codewords = code.encode(messages)
    assert np.all(code.syndrome(codewords) == 0)


def test_weighted_reliability_grand_corrects_single_flip() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=1)
    message = np.random.default_rng(2).integers(0, 2, size=(code.k,), dtype=np.uint8)
    codeword = code.encode(message)[0]
    llr = (1.0 - 2.0 * codeword.astype(np.float32)) * 4.0
    flipped = codeword.copy()
    flipped[3] ^= 1
    llr[3] *= -1.0
    hard_bits = flipped.copy()

    decoder = WeightedReliabilityGRAND(
        code=code,
        pool_size=8,
        max_weight=2,
        budget=100,
        weight_penalties=[0.0, 0.0, 0.1],
    )
    result = decoder.decode(llr=llr, hard_bits=hard_bits, truth_error_mask=(hard_bits ^ codeword))
    assert result.success
    assert np.array_equal(result.decoded_codeword, codeword)
    assert result.queries == result.primary_queries + result.fallback_queries


def test_feature_confidence_label_uses_rank_limit() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=3)
    message = np.random.default_rng(4).integers(0, 2, size=(code.k,), dtype=np.uint8)
    codeword = code.encode(message)[0]
    llr = (1.0 - 2.0 * codeword.astype(np.float32)) * 2.0
    hard_bits = codeword.copy()
    hard_bits[10] ^= 1
    llr[10] *= -1.0
    feat = build_decoder_features(
        llr=llr,
        hard_bits=hard_bits,
        code=code,
        snr_db=2.0,
        profile_id=0,
        num_segments=4,
        error_mask=(hard_bits ^ codeword),
        confidence_weight_threshold=2,
        confidence_rank_limit=4,
    )
    assert feat.confidence_label in (0.0, 1.0)


def test_model_forward_shapes() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=3)
    message = np.random.default_rng(4).integers(0, 2, size=(code.k,), dtype=np.uint8)
    codeword = code.encode(message)[0]
    llr = (1.0 - 2.0 * codeword.astype(np.float32)) * 2.0
    hard_bits = (llr < 0).astype(np.uint8)
    feat = build_decoder_features(
        llr=llr,
        hard_bits=hard_bits,
        code=code,
        snr_db=2.0,
        profile_id=0,
        num_segments=4,
        error_mask=(hard_bits ^ codeword),
    )
    model = PosteriorSchedulerNet(
        n_bits=code.n,
        per_bit_dim=feat.bit_features.shape[-1],
        global_dim=feat.global_features.shape[-1],
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_multiplier=2,
        dropout=0.1,
        num_segments=4,
        max_weight_class=3,
    )
    out = model(torch.from_numpy(feat.bit_features).unsqueeze(0), torch.from_numpy(feat.global_features).unsqueeze(0))
    assert out["bit_logits"].shape == (1, code.n)
    assert out["segment_logits"].shape == (1, 4)
    assert out["weight_logits"].shape == (1, 5)


def test_nsgrand_fallback_accounts_primary_plus_fallback_queries() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=7)
    message = np.random.default_rng(8).integers(0, 2, size=(code.k,), dtype=np.uint8)
    codeword = code.encode(message)[0]
    llr = (1.0 - 2.0 * codeword.astype(np.float32)) * 5.0
    hard_bits = codeword.copy()
    hard_bits[1] ^= 1
    llr[1] *= -1.0

    # The model points the AI search to the wrong rank, so AI exhausts its tiny budget and fallback must fix the error.
    model = DummyModel(n_bits=code.n, num_segments=4, max_weight_class=4, confidence=0.9, overflow_prob=0.1, hot_rank=min(5, code.n - 1))
    fallback = WeightedReliabilityGRAND(
        code=code,
        pool_size=8,
        max_weight=2,
        budget=100,
        weight_penalties=[0.0, 0.0, 0.1],
    )
    decoder = NeuroSymbolicGRAND(
        code=code,
        model=model,
        device="cpu",
        num_segments=4,
        max_weight_class=4,
        ai_pool_size=4,
        ai_max_weight=1,
        ai_budget=1,
        fallback_decoder=fallback,
        ai_weight_penalties=[0.0, 0.0],
        top_segments=1,
        top_bits_extra=0,
        confidence_threshold=0.35,
        candidate_mass_threshold=0.85,
        adaptive_pool_size=6,
        adaptive_max_weight=2,
        adaptive_budget=2,
        adaptive_top_segments=1,
        adaptive_top_bits_extra=0,
        overflow_expand_threshold=0.5,
        overflow_direct_fallback_threshold=0.95,
        confidence_presearch_threshold=0.01,
        always_fallback_after_ai_fail=True,
        trace_top_attempts=10,
    )
    result = decoder.decode(llr=llr, hard_bits=hard_bits, snr_db=2.0, profile_id=0, truth_error_mask=(hard_bits ^ codeword))
    assert result.fallback_used
    assert result.success
    assert result.primary_queries > 0
    assert result.fallback_queries > 0
    assert result.queries == result.primary_queries + result.fallback_queries



def test_nsgrand_skip_hopeless_returns_zero_queries() -> None:
    code = build_systematic_sparse_code(n=32, k=16, p_column_weight=3, seed=9)
    message = np.random.default_rng(10).integers(0, 2, size=(code.k,), dtype=np.uint8)
    codeword = code.encode(message)[0]
    llr = (1.0 - 2.0 * codeword.astype(np.float32)) * 3.0
    hard_bits = codeword.copy()
    hard_bits[2] ^= 1
    llr[2] *= -1.0

    model = DummyModel(n_bits=code.n, num_segments=4, max_weight_class=4, confidence=0.05, overflow_prob=0.95, hot_rank=min(2, code.n - 1))
    fallback = WeightedReliabilityGRAND(
        code=code,
        pool_size=8,
        max_weight=2,
        budget=100,
        weight_penalties=[0.0, 0.0, 0.1],
    )
    decoder = NeuroSymbolicGRAND(
        code=code,
        model=model,
        device="cpu",
        num_segments=4,
        max_weight_class=4,
        ai_pool_size=4,
        ai_max_weight=1,
        ai_budget=1,
        fallback_decoder=fallback,
        ai_weight_penalties=[0.0, 0.0],
        top_segments=1,
        top_bits_extra=0,
        confidence_threshold=0.35,
        candidate_mass_threshold=0.85,
        adaptive_pool_size=6,
        adaptive_max_weight=2,
        adaptive_budget=2,
        adaptive_top_segments=1,
        adaptive_top_bits_extra=0,
        overflow_expand_threshold=0.5,
        overflow_direct_fallback_threshold=0.90,
        confidence_presearch_threshold=0.01,
        overflow_direct_action="skip",
        overflow_direct_confidence_ceiling=0.20,
        always_fallback_after_ai_fail=True,
        trace_top_attempts=10,
    )
    result = decoder.decode(llr=llr, hard_bits=hard_bits, snr_db=2.0, profile_id=0, truth_error_mask=(hard_bits ^ codeword))
    assert not result.success
    assert not result.fallback_used
    assert result.queries == 0
    assert result.stage == "skip_hopeless"
    assert result.gate_reason == "presearch_skip_hopeless"
