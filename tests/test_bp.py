import numpy as np

from hybrid_bp_nsg.codes.peg_ldpc import build_peg_ldpc
from hybrid_bp_nsg.decoders.bp import BeliefPropagationDecoder


def test_bp_noiseless_decodes():
    code = build_peg_ldpc(n=32, k=16, variable_degree=3, seed=456, peg_restarts=10)
    msg = np.zeros(code.k, dtype=np.uint8)
    cw = code.encode(msg)
    llr = (1.0 - 2.0 * cw.astype(np.float32)) * 8.0
    dec = BeliefPropagationDecoder(code, algorithm='spa', max_iters=20)
    res = dec.decode(llr)
    assert res.success
    assert np.array_equal(res.hard, cw)
