from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..utils.math_utils import ebn0_db_to_noise_var


@dataclass
class Frame:
    message: np.ndarray          # transport / payload bits when available
    codeword_internal: np.ndarray
    codeword_tx: np.ndarray
    llr_internal: np.ndarray
    llr_tx: np.ndarray


def simulate_frame(code: LDPCCode, snr_db: float, profile: str = "A", rng: np.random.Generator | None = None) -> Frame:
    rng = rng or np.random.default_rng()
    msg_len = int(getattr(code, "transport_k", code.k))
    msg = rng.integers(0, 2, size=msg_len, dtype=np.uint8)
    c_internal, c_tx = code.encode_payload(msg)
    c_internal = np.asarray(c_internal, dtype=np.uint8).reshape(-1)
    c_tx = np.asarray(c_tx, dtype=np.uint8).reshape(-1)
    x = 1.0 - 2.0 * c_tx.astype(np.float32)
    noise_var = ebn0_db_to_noise_var(float(snr_db), rate=float(code.rate))
    sigma = float(np.sqrt(noise_var))
    profile = str(profile).upper()

    if profile == "A":
        y = x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)
    elif profile == "C":
        h = np.maximum(0.08, rng.rayleigh(scale=1.0, size=x.shape).astype(np.float32))
        y = h * x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * h * y / max(noise_var, 1e-12)
    elif profile == "E":
        impulsive = rng.random(size=x.shape) < 0.03
        local_sigma = sigma * np.where(impulsive, 4.0, 1.0).astype(np.float32)
        y = x + local_sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)
    else:
        y = x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)

    llr_internal = code.tx_llr_to_internal(llr_tx.astype(np.float32))
    return Frame(msg, c_internal, c_tx, llr_internal.astype(np.float32), llr_tx.astype(np.float32))
