import numpy as np
from hybrid_bp_nsg.codes.gf2 import gf2_solve, gf2_nullspace

def test_gf2_solve():
    A=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    b=np.array([1,0],dtype=np.uint8)
    x=gf2_solve(A,b)
    assert x is not None
    assert np.all((A@x)%2==b)

def test_nullspace():
    H=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    G=gf2_nullspace(H)
    assert G.shape[1]==3
    assert np.all((H@G.T)%2==0)
