import numpy as np

from hybrid_bp_nsg.codes.peg_ldpc import build_peg_ldpc
from hybrid_bp_nsg.codes.gf2 import gf2_rank


def test_ldpc_rank_and_generator():
    code = build_peg_ldpc(n=32, k=16, variable_degree=3, seed=123, peg_restarts=10)
    assert gf2_rank(code.h) == code.m
    g = code.g
    assert g.shape == (code.k, code.n)
    assert np.all((code.h @ g.T) % 2 == 0)
