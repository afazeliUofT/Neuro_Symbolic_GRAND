from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .gf2 import gf2_nullspace, gf2_rank


@dataclass
class LDPCCode:
    h: np.ndarray
    k: int
    n: int
    family: str = "generic_ldpc"
    rate: float | None = None
    g: np.ndarray | None = None
    encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None
    rm_pattern: np.ndarray | None = None
    bg: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.h = (np.asarray(self.h, dtype=np.uint8) & 1)
        self.m, self.n = self.h.shape
        self.k = int(self.k)
        self.rate = float(self.rate if self.rate is not None else self.k / max(1, self.n))
        self.deg_v = np.asarray(self.h.sum(axis=0), dtype=np.float32)
        self.deg_c = np.asarray(self.h.sum(axis=1), dtype=np.float32)
        if self.rm_pattern is not None:
            self.rm_pattern = np.asarray(self.rm_pattern, dtype=np.int8)
            if self.rm_pattern.size != self.n:
                raise ValueError("rm_pattern length must equal internal n")
        if self.g is not None:
            self.g = (np.asarray(self.g, dtype=np.uint8) & 1)

    @property
    def tx_positions(self) -> np.ndarray:
        if self.rm_pattern is None:
            return np.arange(self.n, dtype=np.int32)
        return np.flatnonzero(self.rm_pattern == 1).astype(np.int32)

    @property
    def punctured_positions(self) -> np.ndarray:
        if self.rm_pattern is None:
            return np.zeros(0, dtype=np.int32)
        return np.flatnonzero(self.rm_pattern == 0).astype(np.int32)

    @property
    def transmitted_n(self) -> int:
        return int(self.tx_positions.size)

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        return (self.h @ bits.reshape(-1).astype(np.uint8)) % 2

    def is_codeword(self, bits: np.ndarray) -> bool:
        return int(self.syndrome(bits).sum()) == 0

    def encode_internal(self, message: np.ndarray) -> np.ndarray:
        msg = np.asarray(message, dtype=np.uint8)
        squeeze = msg.ndim == 1
        if squeeze:
            msg = msg[None, :]
        if self.encoder is not None:
            out = self.encoder(msg)
            out = (np.asarray(out, dtype=np.uint8) & 1)
            return out[0] if squeeze else out
        if self.g is None:
            self.g = gf2_nullspace(self.h)
            if self.g.shape[0] < self.k:
                raise RuntimeError(f"Nullspace dimension {self.g.shape[0]} < k={self.k}")
            if self.g.shape[0] > self.k:
                self.g = self.g[: self.k]
        out = (msg[:, : self.k].astype(np.uint8) @ self.g[: self.k].astype(np.uint8)) % 2
        return out[0].astype(np.uint8) if squeeze else out.astype(np.uint8)

    def encode(self, message: np.ndarray) -> np.ndarray:
        internal = self.encode_internal(message)
        if internal.ndim == 1:
            return internal[self.tx_positions]
        return internal[:, self.tx_positions]

    def expand_llr(self, llr_tx: np.ndarray, info_llr: np.ndarray | None = None) -> np.ndarray:
        llr_tx = np.asarray(llr_tx, dtype=np.float32)
        squeeze = llr_tx.ndim == 1
        if squeeze:
            llr_tx = llr_tx[None, :]
        out = np.zeros((llr_tx.shape[0], self.n), dtype=np.float32)
        tx_pos = self.tx_positions
        if llr_tx.shape[1] != tx_pos.size:
            raise ValueError(f"expected {tx_pos.size} transmitted LLRs, got {llr_tx.shape[1]}")
        out[:, tx_pos] = llr_tx
        if info_llr is not None:
            info_llr = np.asarray(info_llr, dtype=np.float32)
            if info_llr.ndim == 1:
                info_llr = info_llr[None, :]
            punc = self.punctured_positions
            punc_info = punc[punc < min(self.k, info_llr.shape[1])]
            if punc_info.size:
                out[:, punc_info] = info_llr[:, punc_info]
        return out[0] if squeeze else out

    def info_bits(self, internal_bits: np.ndarray) -> np.ndarray:
        arr = np.asarray(internal_bits, dtype=np.uint8)
        return arr[..., : self.k]


def _random_regular_h(k: int, n: int, dv: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = n - k
    if m <= 0:
        raise ValueError("n must be larger than k")
    best = None
    for attempt in range(200):
        h = np.zeros((m, n), dtype=np.uint8)
        for v in range(n):
            rows = rng.choice(m, size=min(dv, m), replace=False)
            h[rows, v] = 1
        # Ensure no empty checks.
        for r in np.flatnonzero(h.sum(axis=1) == 0):
            h[r, int(rng.integers(0, n))] = 1
        rank = gf2_rank(h)
        if best is None or rank > gf2_rank(best):
            best = h
        if rank == m:
            return h
    return best


def build_peg_ldpc(k: int = 32, n: int = 64, dv: int = 3, dc: int | None = None, seed: int = 1234, **_) -> LDPCCode:
    # This is a compact random regular LDPC builder used for selftests and environments
    # where Sionna is not installed. It is not intended to reproduce a standard code.
    h = _random_regular_h(int(k), int(n), int(dv), int(seed))
    g = gf2_nullspace(h)
    if g.shape[0] < int(k):
        raise RuntimeError(f"generated H has nullspace dimension {g.shape[0]}, expected at least k={k}")
    g = g[: int(k)]
    return LDPCCode(h=h, k=int(k), n=int(n), family="peg_ldpc", rate=float(k) / float(n), g=g)
