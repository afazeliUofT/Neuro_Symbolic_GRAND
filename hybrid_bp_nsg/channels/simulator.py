from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..utils.math_utils import ebn0_db_to_noise_var


@dataclass
class Frame:
    message: np.ndarray
    codeword_internal: np.ndarray
    codeword_tx: np.ndarray
    llr_internal: np.ndarray
    llr_tx: np.ndarray


def simulate_frame(code: LDPCCode, snr_db: float, profile: str = "A", rng: np.random.Generator | None = None) -> Frame:
    rng = rng or np.random.default_rng()
    msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
    c_internal = code.encode_internal(msg).astype(np.uint8)
    c_tx = c_internal[code.tx_positions].astype(np.uint8)
    x = 1.0 - 2.0 * c_tx.astype(np.float32)
    noise_var = ebn0_db_to_noise_var(float(snr_db), rate=code.rate)
    sigma = float(np.sqrt(noise_var))
    profile = str(profile).upper()

    if profile == "A":
        h = np.ones_like(x, dtype=np.float32)
        y = x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)
    elif profile == "C":
        # Coherent Rayleigh-like local fading with known CSI.
        h = np.maximum(0.08, rng.rayleigh(scale=1.0, size=x.shape).astype(np.float32))
        y = h * x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * h * y / max(noise_var, 1e-12)
    elif profile == "E":
        # Mild impulsive mixture; receiver uses nominal variance.
        impulsive = rng.random(size=x.shape) < 0.03
        local_sigma = sigma * np.where(impulsive, 4.0, 1.0).astype(np.float32)
        y = x + local_sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)
    else:
        y = x + sigma * rng.normal(size=x.shape).astype(np.float32)
        llr_tx = 2.0 * y / max(noise_var, 1e-12)

    llr_internal = code.expand_llr(llr_tx.astype(np.float32))
    return Frame(msg, c_internal, c_tx, llr_internal.astype(np.float32), llr_tx.astype(np.float32))
