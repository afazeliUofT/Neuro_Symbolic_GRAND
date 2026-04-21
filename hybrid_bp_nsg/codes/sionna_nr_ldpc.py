from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from .gf2 import syndrome


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(x, attr):
            try:
                if attr == "detach":
                    x = x.detach()
                elif attr == "cpu":
                    x = x.cpu()
                else:
                    return np.asarray(x.numpy())
            except Exception:
                pass
    return np.asarray(x)


def _import_encoder_decoder() -> Tuple[Any, Any]:
    tried = []
    enc_candidates = [
        ("sionna.phy.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("sionna.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("sionna.fec.ldpc", "LDPC5GEncoder"),
    ]
    dec_candidates = [
        ("sionna.phy.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("sionna.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("sionna.fec.ldpc", "LDPC5GDecoder"),
    ]
    Enc = None
    Dec = None
    for mod_name, cls_name in enc_candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            Enc = getattr(mod, cls_name)
            break
        except Exception as e:  # pragma: no cover - import variability
            tried.append(f"{mod_name}.{cls_name}: {e}")
    for mod_name, cls_name in dec_candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            Dec = getattr(mod, cls_name)
            break
        except Exception as e:  # pragma: no cover - import variability
            tried.append(f"{mod_name}.{cls_name}: {e}")
    if Enc is None or Dec is None:
        raise ImportError("Could not import Sionna 5G LDPC encoder/decoder. Tried: " + " | ".join(tried))
    return Enc, Dec


def _instantiate_encoder(k: int, n: int):
    Enc, _ = _import_encoder_decoder()
    tries = [
        dict(k=k, n=n),
        dict(k=k, n=n, num_bits_per_symbol=1),
    ]
    for kwargs in tries:
        try:
            return Enc(**kwargs)
        except TypeError:
            pass
    try:
        return Enc(k, n)
    except TypeError:
        return Enc(k=k, n=n)


def _instantiate_decoder(enc: Any):
    _, Dec = _import_encoder_decoder()
    attempts = [
        dict(encoder=enc, num_iter=1, hard_out=True, return_infobits=False),
        dict(encoder=enc, num_iter=1, hard_out=True),
        dict(encoder=enc, num_iter=1),
    ]
    for kwargs in attempts:
        try:
            return Dec(**kwargs)
        except TypeError:
            pass
        try:
            return Dec(enc, **{k: v for k, v in kwargs.items() if k != 'encoder'})
        except TypeError:
            pass
    return Dec(enc)


def _extract_dense_pcm(dec: Any) -> np.ndarray:
    attr_paths = [
        "pcm", "_pcm", "decoder.pcm", "decoder._pcm", "_decoder.pcm", "_decoder._pcm"
    ]
    for path in attr_paths:
        cur = dec
        ok = True
        for part in path.split('.'):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if not ok:
            continue
        try:
            dense = cur.todense() if hasattr(cur, 'todense') else cur
            arr = _to_numpy(dense)
            if arr.ndim == 2 and arr.size > 0:
                return (arr.astype(np.int64) % 2).astype(np.uint8)
        except Exception:
            continue
    raise AttributeError('Unable to extract dense PCM from LDPC5GDecoder')


def _generate_pruned_pcm_5g(enc: Any, dec: Any, n_tx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the internal 5G-NR LDPC PCM after removing shortened nodes.

    Important: as in NVIDIA's public gnn-decoder reference, the resulting width
    can be larger than the transmitted codeword length `n_tx` because it still
    includes the first `2*z` punctured information bits.
    """
    pcm_dense = _extract_dense_pcm(dec)
    enc_ref = getattr(dec, '_encoder', enc)
    z = int(getattr(enc_ref, 'z'))
    k = int(getattr(enc_ref, 'k'))
    k_ldpc = int(getattr(enc_ref, 'k_ldpc'))
    n_ldpc = int(getattr(enc_ref, 'n_ldpc'))
    nb_pruned = int(getattr(dec, '_nb_pruned_nodes', 0))

    pos_tx = np.ones(int(n_tx), dtype=np.float32)
    pos_punc = np.concatenate([np.zeros([2*z], dtype=np.float32), pos_tx], axis=0)
    k_short = int(k_ldpc - k)
    num_punc_bits = int((n_ldpc - k_short) - int(n_tx) - 2*z)
    if num_punc_bits < nb_pruned:
        raise RuntimeError(
            f'Inconsistent Sionna 5G rate-matching state: num_punc_bits={num_punc_bits}, nb_pruned={nb_pruned}'
        )
    tail_zeros = np.zeros([max(0, num_punc_bits - nb_pruned)], dtype=np.float32)
    pos_punc2 = np.concatenate([pos_punc, tail_zeros], axis=0)
    pos_info = pos_punc2[0:k]
    num_par_bits = int(n_ldpc - k_short - k - nb_pruned)
    pos_parity = pos_punc2[k:k+num_par_bits]
    pos_short = 2*np.ones([k_short], dtype=np.float32)
    rm_pattern = np.concatenate([pos_info, pos_short, pos_parity], axis=0)
    pcm_pruned = np.copy(pcm_dense)
    idx_short = np.where(rm_pattern == 2)[0]
    idx_keep = np.setdiff1d(np.arange(pcm_pruned.shape[1]), idx_short)
    pcm_pruned = pcm_pruned[:, idx_keep]
    rm_pattern_pruned = rm_pattern[idx_keep]
    return (pcm_pruned.astype(np.int64) % 2).astype(np.uint8), rm_pattern_pruned.astype(np.int8)


@dataclass
class SionnaNRLDPCCode:
    encoder: Any
    decoder: Any
    h: np.ndarray
    rm_pattern: np.ndarray
    n: int  # internal graph width (includes punctured systematic bits)
    transmitted_n: int  # actual transmitted length after rate-matching
    k: int
    m: int
    seed: int
    family: str
    bg: str
    requested_n: int
    requested_k: int
    edge_vars: np.ndarray
    edge_checks: np.ndarray
    var_edges: List[np.ndarray]
    check_edges: List[np.ndarray]
    deg_v: np.ndarray
    deg_c: np.ndarray

    @property
    def rate(self) -> float:
        return self.k / self.transmitted_n

    @property
    def tx_positions(self) -> np.ndarray:
        return np.flatnonzero(self.rm_pattern == 1).astype(np.int32)

    @property
    def punctured_positions(self) -> np.ndarray:
        return np.flatnonzero(self.rm_pattern == 0).astype(np.int32)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode to the transmitted rate-matched codeword of length transmitted_n."""
        msg = np.asarray(message, dtype=np.float32)
        if msg.ndim == 1:
            msg = msg[None, :]
        try:
            import tensorflow as tf
            x = tf.convert_to_tensor(msg, dtype=tf.float32)
            c = self.encoder(x)
            out = _to_numpy(c)
        except Exception:
            out = _to_numpy(self.encoder(msg))
        out = (np.asarray(out) > 0.5).astype(np.uint8)
        if out.ndim == 1:
            return out
        return out if message.ndim == 2 else out[0]

    def encode_internal(self, message: np.ndarray) -> np.ndarray:
        tx = self.encode(message)
        msg = np.asarray(message, dtype=np.uint8)
        if msg.ndim == 1:
            msg = msg[None, :]
            squeeze = True
        else:
            squeeze = False
        tx = np.asarray(tx, dtype=np.uint8)
        if tx.ndim == 1:
            tx = tx[None, :]
        out = np.zeros((msg.shape[0], self.n), dtype=np.uint8)
        # The first k positions correspond to the systematic information bits.
        out[:, :self.k] = msg
        out[:, self.tx_positions] = tx
        # ensure punctured positions inside the first k stay equal to message bits
        out[:, self.punctured_positions] = out[:, self.punctured_positions]
        return out[0] if squeeze else out

    def expand_llr(self, llr_tx: np.ndarray, info_llr: np.ndarray | None = None) -> np.ndarray:
        llr_tx = np.asarray(llr_tx, dtype=np.float32)
        if llr_tx.ndim == 1:
            llr_tx = llr_tx[None, :]
            squeeze = True
        else:
            squeeze = False
        out = np.zeros((llr_tx.shape[0], self.n), dtype=np.float32)
        out[:, self.tx_positions] = llr_tx
        if info_llr is not None:
            info_llr = np.asarray(info_llr, dtype=np.float32)
            if info_llr.ndim == 1:
                info_llr = info_llr[None, :]
            out[:, self.punctured_positions] = info_llr[:, self.punctured_positions]
        return out[0] if squeeze else out

    def info_from_internal(self, word: np.ndarray) -> np.ndarray:
        arr = np.asarray(word, dtype=np.uint8)
        return arr[..., :self.k]

    def syndrome(self, word: np.ndarray) -> np.ndarray:
        return syndrome(self.h, word)


def _build_edges(h: np.ndarray):
    m, n = h.shape
    edge_vars=[]
    edge_checks=[]
    var_edges=[[] for _ in range(n)]
    check_edges=[[] for _ in range(m)]
    e=0
    for c in range(m):
        for v in np.flatnonzero(h[c]):
            edge_vars.append(v)
            edge_checks.append(c)
            var_edges[v].append(e)
            check_edges[c].append(e)
            e += 1
    return (
        np.array(edge_vars, dtype=np.int32),
        np.array(edge_checks, dtype=np.int32),
        [np.array(x, dtype=np.int32) for x in var_edges],
        [np.array(x, dtype=np.int32) for x in check_edges],
        np.array([len(x) for x in var_edges], dtype=np.int32),
        np.array([len(x) for x in check_edges], dtype=np.int32),
    )


def build_sionna_nr_ldpc(n: int, k: int, seed: int = 31415, align_to_pcm_length: bool = True,
                         strict_pcm_check: bool = True, **_: object) -> SionnaNRLDPCCode:
    requested_n = int(n)
    requested_k = int(k)
    enc = _instantiate_encoder(requested_k, requested_n)
    dec = _instantiate_decoder(enc)
    h, rm_pattern = _generate_pruned_pcm_5g(enc, dec, requested_n)
    internal_n = int(h.shape[1])
    tx_count = int(np.sum(rm_pattern == 1))
    if tx_count != requested_n:
        raise RuntimeError(
            f'Internal 5G NR mapping mismatch: rm_pattern marks {tx_count} transmitted positions, expected requested n={requested_n}'
        )
    # Validate on the all-zero codeword using internal reconstruction.
    try:
        c0_tx = _to_numpy(enc(np.zeros((1, requested_k), dtype=np.float32))).reshape(-1)
        c0_tx = (c0_tx > 0.5).astype(np.uint8)
        if c0_tx.size != requested_n:
            raise RuntimeError(f'Sionna encoder output length {c0_tx.size} does not match requested transmitted n={requested_n}')
        c0_internal = np.zeros(internal_n, dtype=np.uint8)
        c0_internal[np.flatnonzero(rm_pattern == 1)] = c0_tx
        syn = (h @ c0_internal) % 2
        if np.any(syn != 0):
            raise RuntimeError('Internal 5G NR LDPC PCM does not validate the reconstructed all-zero codeword.')
    except Exception:
        if strict_pcm_check:
            raise
    edge_vars, edge_checks, var_edges, check_edges, deg_v, deg_c = _build_edges(h)
    bg = 'unknown'
    for attr in ('bg', '_bg', 'base_graph', '_base_graph'):
        if hasattr(enc, attr):
            try:
                bg = str(getattr(enc, attr))
                break
            except Exception:
                pass
    return SionnaNRLDPCCode(
        encoder=enc,
        decoder=dec,
        h=h.astype(np.uint8),
        rm_pattern=rm_pattern,
        n=int(h.shape[1]),
        transmitted_n=requested_n,
        k=requested_k,
        m=int(h.shape[0]),
        seed=int(seed),
        family='sionna_nr_ldpc',
        bg=bg,
        requested_n=int(n),
        requested_k=int(k),
        edge_vars=edge_vars,
        edge_checks=edge_checks,
        var_edges=var_edges,
        check_edges=check_edges,
        deg_v=deg_v,
        deg_c=deg_c,
    )


def code_summary(code: SionnaNRLDPCCode) -> Dict[str, int | float | str]:
    return {
        'family': code.family,
        'requested_n': int(code.requested_n),
        'requested_k': int(code.requested_k),
        'n_internal': int(code.n),
        'n_transmitted': int(code.transmitted_n),
        'k': int(code.k),
        'm': int(code.m),
        'rate': float(code.rate),
        'edges': int(code.edge_vars.size),
        'avg_check_degree': float(code.deg_c.mean()) if code.deg_c.size else 0.0,
        'max_check_degree': int(code.deg_c.max()) if code.deg_c.size else 0,
        'min_check_degree': int(code.deg_c.min()) if code.deg_c.size else 0,
        'avg_variable_degree': float(code.deg_v.mean()) if code.deg_v.size else 0.0,
        'num_punctured_internal_positions': int(np.sum(code.rm_pattern == 0)),
        'num_transmitted_internal_positions': int(np.sum(code.rm_pattern == 1)),
        'bg': code.bg,
    }
