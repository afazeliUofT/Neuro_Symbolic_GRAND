from __future__ import annotations

import numpy as np

from neuro_symbolic_grand.codes import build_systematic_sparse_code
from neuro_symbolic_grand.decoders import WeightedReliabilityGRAND
from neuro_symbolic_grand.decoders.features import build_decoder_features
from neuro_symbolic_grand.models.posterior_scorer import PosteriorSchedulerNet


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
    import torch
    out = model(torch.from_numpy(feat.bit_features).unsqueeze(0), torch.from_numpy(feat.global_features).unsqueeze(0))
    assert out["bit_logits"].shape == (1, code.n)
    assert out["segment_logits"].shape == (1, 4)
    assert out["weight_logits"].shape == (1, 5)
