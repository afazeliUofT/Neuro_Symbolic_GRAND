from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..codes.systematic_sparse import SystematicSparseCode
from ..utils.math_utils import next_power_of_two
from .factory import build_channel_backend


@dataclass
class ChannelSimulationContext:
    code: SystematicSparseCode
    channel_cfg: Dict[str, object]
    seed: int
    tf_threads: int = 1

    def __post_init__(self) -> None:
        self.n_fft = max(next_power_of_two(self.code.n), self.code.n)
        self.active_bins = np.arange(self.code.n, dtype=np.int64)
        self.rng = np.random.default_rng(self.seed)
        self.backends: Dict[str, object] = {}

    def _get_backend(self, profile: str):
        if profile not in self.backends:
            self.backends[profile] = build_channel_backend(
                channel_cfg=self.channel_cfg,
                profile=profile,
                n_fft=self.n_fft,
                seed=int(self.rng.integers(0, 2**31 - 1)),
                tf_threads=self.tf_threads,
            )
        return self.backends[profile]

    def simulate_batch(
        self,
        batch_size: int,
        profiles: List[str],
        snr_db_values: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if len(profiles) != batch_size:
            raise ValueError("profiles length must match batch size")
        snr_db_values = np.asarray(snr_db_values, dtype=np.float32)
        if snr_db_values.shape != (batch_size,):
            raise ValueError("snr_db_values must have shape [batch_size]")

        messages = self.rng.integers(0, 2, size=(batch_size, self.code.k), dtype=np.uint8)
        codewords = self.code.encode(messages)
        bpsk = (1.0 - 2.0 * codewords.astype(np.float32)).astype(np.complex64)

        h_active = np.zeros((batch_size, self.code.n), dtype=np.complex64)
        for profile in sorted(set(profiles)):
            mask = np.array([p == profile for p in profiles], dtype=bool)
            num = int(mask.sum())
            backend = self._get_backend(profile)
            try:
                h_full = backend.generate_frequency_response(num)
            except Exception:
                if str(self.channel_cfg.get("backend", "sionna_tdl")) == "sionna_tdl" and bool(self.channel_cfg.get("allow_fallback", True)):
                    fallback_cfg = dict(self.channel_cfg)
                    fallback_cfg["backend"] = "fallback_tdl"
                    backend = build_channel_backend(
                        channel_cfg=fallback_cfg,
                        profile=profile,
                        n_fft=self.n_fft,
                        seed=int(self.rng.integers(0, 2**31 - 1)),
                        tf_threads=1,
                    )
                    self.backends[profile] = backend
                    h_full = backend.generate_frequency_response(num)
                else:
                    raise
            h_active[mask] = h_full[:, self.active_bins]

        no = (10.0 ** (-snr_db_values / 10.0)).astype(np.float32)
        noise = (
            self.rng.normal(size=(batch_size, self.code.n))
            + 1j * self.rng.normal(size=(batch_size, self.code.n))
        ).astype(np.complex64) * np.sqrt(no[:, None] / 2.0).astype(np.float32)
        y = h_active * bpsk + noise
        matched = np.conj(h_active) * y
        llr = 2.0 * matched.real / np.maximum(no[:, None], 1e-8)
        hard_bits = (llr < 0.0).astype(np.uint8)
        error_mask = self.code.hard_error_mask(hard_bits, codewords)
        syndrome = self.code.syndrome(hard_bits).astype(np.uint8)
        return {
            "messages": messages,
            "codewords": codewords,
            "hard_bits": hard_bits,
            "error_mask": error_mask,
            "llr": llr.astype(np.float32),
            "snr_db": snr_db_values.astype(np.float32),
            "h_freq_real": h_active.real.astype(np.float32),
            "h_freq_imag": h_active.imag.astype(np.float32),
            "syndrome": syndrome,
        }
