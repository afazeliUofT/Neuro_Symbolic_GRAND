from hybrid_bp_nsg.codes import sionna_nr_ldpc as m
import numpy as np


class DummyEnc:
    def __init__(self):
        self.z = 2
        self.k = 10
        self.k_ldpc = 12
        self.n_ldpc = 20
        self._bg = 'bg1'
    def __call__(self, x):
        return np.zeros((x.shape[0], 16), dtype=float)


class DummyDec:
    _nb_pruned_nodes = 0


def test_internal_width_can_exceed_transmitted_n(monkeypatch):
    monkeypatch.setattr(m, '_instantiate_encoder', lambda k, n: DummyEnc())
    monkeypatch.setattr(m, '_instantiate_decoder', lambda enc: DummyDec())
    h = np.zeros((4, 18), dtype='uint8')
    rm = np.array([0,0] + [1]*16, dtype='int8')
    monkeypatch.setattr(m, '_generate_pruned_pcm_5g', lambda enc, dec, n: (h, rm))
    monkeypatch.setattr(m, '_build_edges', lambda h: (np.array([],dtype='int32'), np.array([],dtype='int32'), [], [], np.array([],dtype='int32'), np.array([],dtype='int32')))
    code = m.build_sionna_nr_ldpc(n=16, k=10, align_to_pcm_length=True, strict_pcm_check=True)
    assert code.transmitted_n == 16
    assert code.n == 18
    assert int((code.rm_pattern == 1).sum()) == 16


def test_encode_internal_inserts_punctured_info_bits(monkeypatch):
    monkeypatch.setattr(m, '_instantiate_encoder', lambda k, n: DummyEnc())
    monkeypatch.setattr(m, '_instantiate_decoder', lambda enc: DummyDec())
    h = np.zeros((4, 18), dtype='uint8')
    rm = np.array([0,0] + [1]*16, dtype='int8')
    monkeypatch.setattr(m, '_generate_pruned_pcm_5g', lambda enc, dec, n: (h, rm))
    monkeypatch.setattr(m, '_build_edges', lambda h: (np.array([],dtype='int32'), np.array([],dtype='int32'), [], [], np.array([],dtype='int32'), np.array([],dtype='int32')))
    code = m.build_sionna_nr_ldpc(n=16, k=10, strict_pcm_check=False)
    msg = np.array([1,0,1,0,1,1,0,0,1,0], dtype=np.uint8)
    internal = code.encode_internal(msg)
    assert internal.shape[0] == 18
    assert np.array_equal(internal[:2], msg[:2])
    assert int((internal == 1).sum()) >= int(msg[:2].sum())
