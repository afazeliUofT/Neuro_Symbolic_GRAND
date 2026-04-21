from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_TDL_POWER_DB = {
    "A": np.array([-13.4, 0.0, -2.2, -4.0, -6.0], dtype=np.float32),
    "C": np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.0], dtype=np.float32),
    "E": np.array([-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8], dtype=np.float32),
}

_TDL_DELAY_NORM = {
    "A": np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610], dtype=np.float32),
    "C": np.array([0.0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584], dtype=np.float32),
    "E": np.array([0.0, 0.0, 0.5133, 0.5440, 0.5630, 0.5630, 0.5630, 0.6584, 0.7000], dtype=np.float32),
}


@dataclass
class LocalFrequencySelectiveBackend:
    profile: str
    delay_spread_s: float
    subcarrier_spacing_hz: float
    n_fft: int
    carrier_frequency_hz: float
    min_speed_mps: float
    max_speed_mps: float
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.sample_rate_hz = self.subcarrier_spacing_hz * self.n_fft
        self.frequencies_hz = np.fft.fftfreq(self.n_fft, d=1.0 / self.sample_rate_hz)
        p_db = _TDL_POWER_DB.get(self.profile, _TDL_POWER_DB["A"])
        self.power_lin = 10.0 ** (p_db / 10.0)
        self.power_lin = self.power_lin / np.sum(self.power_lin)
        self.tau = _TDL_DELAY_NORM.get(self.profile, _TDL_DELAY_NORM["A"]) * float(self.delay_spread_s)

    def generate_frequency_response(self, batch_size: int) -> np.ndarray:
        gains = (
            self.rng.normal(size=(batch_size, self.power_lin.size))
            + 1j * self.rng.normal(size=(batch_size, self.power_lin.size))
        ).astype(np.complex64) * np.sqrt(self.power_lin[None, :] / 2.0)
        exponents = np.exp(-1j * 2.0 * np.pi * self.tau[None, :, None] * self.frequencies_hz[None, None, :])
        h_freq = np.sum(gains[:, :, None] * exponents, axis=1)
        power = np.mean(np.abs(h_freq) ** 2, axis=1, keepdims=True)
        h_freq = h_freq / np.sqrt(np.maximum(power, 1e-12))
        return h_freq.astype(np.complex64)
