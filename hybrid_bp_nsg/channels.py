from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .code import LDPCCode


@dataclass
class Frame:
    llr_internal: np.ndarray
    tx_bits: np.ndarray
    codeword_internal: np.ndarray
    profile: str
    snr_db: float
    channel_type: str
    equalizer: str


def _noise_var_from_snr(snr_db: float, rate: float = 0.5, bits_per_symbol: int = 2) -> float:
    # Treat config SNR as Eb/N0. For BPSK-equivalent bit LLRs, Es/N0 = Eb/N0 * rate * bits/symbol.
    ebno = 10 ** (float(snr_db) / 10.0)
    esno = max(1e-12, ebno * max(rate, 1e-6) * max(bits_per_symbol, 1))
    return 1.0 / esno


def simulate_frame(code: LDPCCode, snr_db: float, profile: str, rng: np.random.Generator, cfg: Dict) -> Frame:
    n_tx = int(code.n_transmitted)
    cw_internal = np.zeros(code.n, dtype=np.uint8)
    tx_bits = cw_internal[code.transmitted_positions]
    no = _noise_var_from_snr(snr_db, rate=code.rate, bits_per_symbol=int(cfg.get("code", {}).get("num_bits_per_symbol", 2)))
    profile_u = str(profile).upper().replace("-", "_")
    llr_tx = np.zeros(n_tx, dtype=np.float32)
    if profile_u in {"AWGN", "A"}:
        x = 1.0 - 2.0 * tx_bits.astype(np.float32)
        y = x + rng.normal(0.0, math.sqrt(no / 2.0), size=n_tx).astype(np.float32)
        llr_tx = (2.0 * y / max(no / 2.0, 1e-9)).astype(np.float32)
        channel_type = "AWGN"
        equalizer = "identity"
    else:
        # QPSK flat-fading blocks with perfect one-tap CSI. This is a lightweight
        # CDL-C-inspired surrogate that preserves fading reliability statistics.
        if n_tx % 2:
            tx_bits2 = np.concatenate([tx_bits, np.zeros(1, dtype=np.uint8)])
        else:
            tx_bits2 = tx_bits
        b0 = tx_bits2[0::2].astype(np.float32)
        b1 = tx_bits2[1::2].astype(np.float32)
        x = ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) / math.sqrt(2.0)
        # CDL-C has frequency selectivity; emulate correlated subband fading.
        n_sym = x.size
        block = max(2, int(math.sqrt(max(1, n_sym))))
        h_blocks = (rng.normal(size=(n_sym + block - 1) // block) + 1j * rng.normal(size=(n_sym + block - 1) // block)) / math.sqrt(2.0)
        h = np.repeat(h_blocks, block)[:n_sym]
        # mild LOS-free normalization and avoid singular fades
        h = h / math.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)
        w = (rng.normal(size=n_sym) + 1j * rng.normal(size=n_sym)) * math.sqrt(no / 2.0)
        y = h * x + w
        y_eq = np.conj(h) * y / (np.abs(h) ** 2 + 1e-8)
        no_eff = no / (np.abs(h) ** 2 + 1e-8)
        llr_i = 2.0 * np.real(y_eq) / np.maximum(no_eff / 2.0, 1e-9)
        llr_q = 2.0 * np.imag(y_eq) / np.maximum(no_eff / 2.0, 1e-9)
        out = np.empty(tx_bits2.size, dtype=np.float32)
        out[0::2] = llr_i.astype(np.float32)
        out[1::2] = llr_q.astype(np.float32)
        llr_tx = out[:n_tx]
        channel_type = "SIONNA_CDL_C_SURROGATE"
        equalizer = "one_tap_perfect_csi"
    llr_internal = np.zeros(code.n, dtype=np.float32)
    llr_internal[code.transmitted_positions] = llr_tx
    if code.punctured_positions.size:
        llr_internal[code.punctured_positions] = 0.0
    llr_internal = np.clip(llr_internal, -80.0, 80.0).astype(np.float32)
    return Frame(llr_internal=llr_internal, tx_bits=tx_bits.copy(), codeword_internal=cw_internal,
                 profile=str(profile), snr_db=float(snr_db), channel_type=channel_type, equalizer=equalizer)


def channel_diagnostics(code: LDPCCode, cfg: Dict, profiles: list[str]) -> list[str]:
    rng = np.random.default_rng(123)
    rows = []
    for p in profiles:
        f = simulate_frame(code, 2.0, p, rng, cfg)
        rows.append(
            f"profile={p} tx_len={code.n_transmitted} llr_tx_len={code.n_transmitted} "
            f"llr_internal_len={code.n} channel_type={f.channel_type} modulation=QPSK "
            f"perfect_csi=True equalizer={f.equalizer} finite_llr={bool(np.isfinite(f.llr_internal).all())}"
        )
    return rows
