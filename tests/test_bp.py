import numpy as np
from hybrid_bp_nsg.codes.peg_ldpc import build_peg_ldpc
from hybrid_bp_nsg.decoders.bp import BeliefPropagationDecoder

def test_bp_clean_codeword():
    code=build_peg_ldpc(k=16,n=32,seed=5)
    msg=np.zeros(code.k,dtype=np.uint8)
    c=code.encode_internal(msg)
    llr=8*(1-2*c.astype(np.float32))
    dec=BeliefPropagationDecoder(code,max_iters=5)
    r=dec.decode(llr)
    assert r.success
    assert np.all(r.hard==c)
