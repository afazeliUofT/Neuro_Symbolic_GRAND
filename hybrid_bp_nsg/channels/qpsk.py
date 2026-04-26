from __future__ import annotations

import numpy as np


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Map bits to Gray-coded QPSK with Es=1.

    Mapping uses independent I/Q bits:
      b0 -> real sign, b1 -> imag sign
      x = ((1-2*b0) + j*(1-2*b1)) / sqrt(2)
    """
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size % 2 != 0:
        raise ValueError(f"QPSK requires an even number of bits, got {arr.size}")
    pairs = arr.reshape(-1, 2).astype(np.float32)
    syms = ((1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])) / np.sqrt(2.0)
    return syms.astype(np.complex64)


def qpsk_to_llr(y: np.ndarray, noise_var: float, h: np.ndarray | None = None) -> np.ndarray:
    """Soft demapper for Gray-coded QPSK.

    Parameters
    ----------
    y : complex ndarray
        Received symbols.
    noise_var : float
        Complex noise variance N0, i.e., E[|n|^2].
    h : complex ndarray or None
        Perfect CSI per symbol. If provided, one-tap equalization is applied via
        the sufficient statistic conj(h)*y. For scalar complex AWGN, set h=None.
    """
    y = np.asarray(y, dtype=np.complex64).reshape(-1)
    if h is None:
        r = y
    else:
        h = np.asarray(h, dtype=np.complex64).reshape(-1)
        if h.size != y.size:
            raise ValueError(f"Expected CSI of length {y.size}, got {h.size}")
        r = np.conj(h) * y
    scale = float(2.0 * np.sqrt(2.0) / max(float(noise_var), 1e-12))
    llr_i = scale * np.real(r)
    llr_q = scale * np.imag(r)
    out = np.empty(2 * y.size, dtype=np.float32)
    out[0::2] = llr_i.astype(np.float32)
    out[1::2] = llr_q.astype(np.float32)
    return out
