from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


_FALLBACK_TDL_PROFILES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    # Approximate fallback profiles used only when Sionna execution is not available.
    # These are not intended to be standards-exact replacements.
    "A": (
        np.array([0.0, 0.3, 0.7, 1.1, 1.9, 2.5], dtype=np.float64),
        np.array([0.0, -2.2, -4.0, -6.0, -8.2, -10.5], dtype=np.float64),
    ),
    "B": (
        np.array([0.0, 0.2, 0.5, 1.0, 1.6, 2.6, 3.9], dtype=np.float64),
        np.array([0.0, -1.5, -3.2, -5.2, -7.5, -9.2, -11.3], dtype=np.float64),
    ),
    "C": (
        np.array([0.0, 0.25, 0.8, 1.6, 2.8, 4.1, 5.8, 7.9], dtype=np.float64),
        np.array([0.0, -1.0, -2.4, -4.5, -6.8, -9.0, -11.8, -14.5], dtype=np.float64),
    ),
    "D": (
        np.array([0.0, 0.1, 0.4, 1.0, 2.0, 3.3, 5.5, 8.0], dtype=np.float64),
        np.array([2.0, -1.0, -3.8, -6.5, -8.5, -11.5, -14.5, -18.0], dtype=np.float64),
    ),
    "E": (
        np.array([0.0, 0.15, 0.55, 1.4, 2.7, 4.8, 7.6, 10.5], dtype=np.float64),
        np.array([3.0, -0.5, -3.0, -6.0, -9.2, -12.0, -15.8, -19.0], dtype=np.float64),
    ),
}


@dataclass
class FallbackTDLBackend:
    profile: str
    delay_spread_s: float
    subcarrier_spacing_hz: float
    n_fft: int
    carrier_frequency_hz: float
    min_speed_mps: float
    max_speed_mps: float
    seed: int = 0

    def __post_init__(self) -> None:
        if self.profile not in _FALLBACK_TDL_PROFILES:
            raise ValueError(f"Unsupported fallback profile {self.profile!r}")
        self.rng = np.random.default_rng(self.seed)
        self.sample_rate_hz = self.subcarrier_spacing_hz * self.n_fft
        self.frequencies_hz = np.fft.fftfreq(self.n_fft, d=1.0 / self.sample_rate_hz)
        delay_shape, powers_db = _FALLBACK_TDL_PROFILES[self.profile]
        self.tau_s = delay_shape * float(self.delay_spread_s)
        powers_lin = 10.0 ** (powers_db / 10.0)
        self.path_gains_std = np.sqrt(powers_lin / powers_lin.sum())

    def generate_frequency_response(self, batch_size: int) -> np.ndarray:
        num_paths = len(self.tau_s)
        real = self.rng.normal(size=(batch_size, num_paths))
        imag = self.rng.normal(size=(batch_size, num_paths))
        taps = (real + 1j * imag) / np.sqrt(2.0)
        taps = taps * self.path_gains_std[None, :]
        # Mild random phase rotation approximating Doppler-induced variability.
        phase_rate = self.rng.uniform(self.min_speed_mps, max(self.max_speed_mps, self.min_speed_mps + 1e-6), size=(batch_size, 1))
        phase = np.exp(1j * 2.0 * np.pi * phase_rate * self.rng.uniform(0.0, 1.0, size=(batch_size, num_paths)) * 1e-3)
        taps = taps * phase
        exponents = np.exp(-1j * 2.0 * np.pi * self.tau_s[None, :, None] * self.frequencies_hz[None, None, :])
        h_freq = np.sum(taps[:, :, None] * exponents, axis=1)
        return h_freq.astype(np.complex64)
