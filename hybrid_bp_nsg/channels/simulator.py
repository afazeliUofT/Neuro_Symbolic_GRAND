from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..utils.math_utils import ebn0_db_to_noise_var
from .qpsk import bits_to_qpsk, qpsk_to_llr


@dataclass
class Frame:
    message: np.ndarray          # transport / payload bits when available
    codeword_internal: np.ndarray
    codeword_tx: np.ndarray
    llr_internal: np.ndarray
    llr_tx: np.ndarray
    channel_meta: Dict[str, Any] = field(default_factory=dict)


def _awgn_qpsk_llr(c_tx: np.ndarray, noise_var: float, rng: np.random.Generator) -> tuple[np.ndarray, Dict[str, Any]]:
    syms = bits_to_qpsk(c_tx)
    sigma = np.sqrt(max(float(noise_var), 1e-12) / 2.0)
    noise = (sigma * rng.normal(size=syms.shape) + 1j * sigma * rng.normal(size=syms.shape)).astype(np.complex64)
    y = syms + noise
    llr = qpsk_to_llr(y, float(noise_var), h=None)
    return llr.astype(np.float32), {
        "channel_type": "AWGN",
        "modulation": "QPSK",
        "perfect_csi": True,
        "equalizer": "identity",
    }


def _cdl_c_qpsk_llr(c_tx: np.ndarray, noise_var: float, rng: np.random.Generator, channel_cfg: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    from .sionna_cdl import simulate_cdl_c_qpsk

    llr, meta = simulate_cdl_c_qpsk(
        tx_bits=np.asarray(c_tx, dtype=np.uint8),
        noise_var=float(noise_var),
        seed=int(rng.integers(0, 2**31 - 1)),
        carrier_frequency_hz=float(channel_cfg.get("carrier_frequency_hz", 3.5e9)),
        subcarrier_spacing_hz=float(channel_cfg.get("subcarrier_spacing_hz", 30e3)),
        num_ofdm_symbols=int(channel_cfg.get("num_ofdm_symbols", 14)),
        fft_size=int(channel_cfg.get("fft_size", 72)),
        cyclic_prefix_length=int(channel_cfg.get("cyclic_prefix_length", 0)),
        delay_spread_s=float(channel_cfg.get("delay_spread_s", 100e-9)),
        speed_m_per_s=float(channel_cfg.get("speed_m_per_s", 0.0)),
        normalize_channel=bool(channel_cfg.get("normalize_channel", True)),
        direction=str(channel_cfg.get("direction", "uplink")),
        model=str(channel_cfg.get("model", "C")),
        perfect_csi=bool(channel_cfg.get("perfect_csi", True)),
    )
    meta = dict(meta)
    meta.setdefault("modulation", "QPSK")
    return llr.astype(np.float32), meta


def simulate_frame(
    code: LDPCCode,
    snr_db: float,
    profile: str = "AWGN",
    rng: np.random.Generator | None = None,
    channel_cfg: Dict[str, Any] | None = None,
) -> Frame:
    """Simulate one coded frame.

    Supported profiles in v13:
      - AWGN: QPSK over complex AWGN
      - CDL_C: QPSK over a Sionna-generated 3GPP CDL-C OFDM channel with one-tap
        perfect-CSI equalization back to bit LLRs.

    Legacy aliases A/C/E are mapped conservatively to AWGN for backward compatibility,
    but the default configs use only AWGN and CDL_C.
    """
    rng = rng or np.random.default_rng()
    channel_cfg = dict(channel_cfg or {})
    msg_len = int(getattr(code, "transport_k", code.k))
    msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
    c_internal, c_tx = code.encode_payload(msg)
    c_internal = np.asarray(c_internal, dtype=np.uint8).reshape(-1)
    c_tx = np.asarray(c_tx, dtype=np.uint8).reshape(-1)
    if c_tx.size % 2 != 0:
        raise RuntimeError(f"QPSK channel expects even transmitted length, got {c_tx.size}")

    noise_var = ebn0_db_to_noise_var(float(snr_db), rate=float(code.rate))
    p = str(profile).upper()
    if p in {"A", "AWGN", "AWGN_QPSK"}:
        llr_tx, meta = _awgn_qpsk_llr(c_tx, noise_var, rng)
    elif p in {"CDL_C", "CDLC", "CDL-C", "C"}:
        llr_tx, meta = _cdl_c_qpsk_llr(c_tx, noise_var, rng, channel_cfg.get("cdl_c", {}))
    elif p in {"E", "IMPULSIVE"}:
        # User requested only AWGN and Sionna CDL-C; keep a backward-compatible alias
        # but map it to AWGN_QPSK rather than the old impulsive-noise profile.
        llr_tx, meta = _awgn_qpsk_llr(c_tx, noise_var, rng)
        meta["profile_alias"] = p
    else:
        raise ValueError(f"Unknown channel profile '{profile}'. Supported profiles: AWGN, CDL_C")

    llr_internal = code.tx_llr_to_internal(llr_tx.astype(np.float32))
    meta.update({
        "snr_db": float(snr_db),
        "noise_var": float(noise_var),
        "profile": p,
    })
    return Frame(msg, c_internal, c_tx, llr_internal.astype(np.float32), llr_tx.astype(np.float32), meta)
