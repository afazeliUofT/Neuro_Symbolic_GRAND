import numpy as np
from hybrid_bp_nsg.codes.peg_ldpc import build_peg_ldpc
from hybrid_bp_nsg.decoders.bp import BPDecodeResult
from hybrid_bp_nsg.training.features import make_grand_base

def test_grand_base_channel_basis():
    code=build_peg_ldpc(k=16,n=32,seed=7)
    llr=np.ones(code.n,dtype=np.float32)
    llr[[1,3]]=-1
    bp=BPDecodeResult(False, np.zeros(code.n,dtype=np.uint8), llr.copy(), np.ones(code.m,dtype=np.uint8), 5, None)
    base=make_grand_base(code,llr,bp,"channel_with_bp_punctures")
    assert base[1]==1 and base[3]==1
