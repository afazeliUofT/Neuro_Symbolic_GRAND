from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..utils.env import configure_tensorflow_threads


@dataclass
class SionnaTDLBackend:
    profile: str
    delay_spread_s: float
    subcarrier_spacing_hz: float
    n_fft: int
    carrier_frequency_hz: float
    min_speed_mps: float
    max_speed_mps: float
    normalization: bool = True
    tf_intra_threads: int = 1
    tf_inter_threads: int = 1

    def __post_init__(self) -> None:
        self.sample_rate_hz = self.subcarrier_spacing_hz * self.n_fft
        self.frequencies_hz = np.fft.fftfreq(self.n_fft, d=1.0 / self.sample_rate_hz)
        self._tdl = None
        self._tf = None

    def _import_backend(self) -> None:
        if self._tdl is not None:
            return
        configure_tensorflow_threads(self.tf_intra_threads, self.tf_inter_threads)
        tf = importlib.import_module("tensorflow")
        try:
            tr38901 = importlib.import_module("sionna.phy.channel.tr38901")
            TDL = getattr(tr38901, "TDL")
        except (ModuleNotFoundError, AttributeError):
            tr38901_tdl = importlib.import_module("sionna.phy.channel.tr38901.tdl")
            TDL = getattr(tr38901_tdl, "TDL")
        self._tf = tf
        self._tdl = TDL(
            model=self.profile,
            delay_spread=float(self.delay_spread_s),
            carrier_frequency=float(self.carrier_frequency_hz),
            min_speed=float(self.min_speed_mps),
            max_speed=float(self.max_speed_mps),
        )

    @staticmethod
    def _squeeze_path_tensor(arr: np.ndarray, batch_size: int, time_axis_last: bool) -> np.ndarray:
        arr = np.asarray(arr)
        if time_axis_last and arr.ndim >= 1:
            arr = np.take(arr, 0, axis=-1)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim > 2:
            if arr.shape[0] == batch_size:
                arr = arr.reshape(batch_size, -1)
            elif arr.shape[-1] == batch_size:
                arr = np.moveaxis(arr, -1, 0).reshape(batch_size, -1)
            else:
                arr = arr.reshape(batch_size, -1)
        return arr

    def generate_frequency_response(self, batch_size: int) -> np.ndarray:
        self._import_backend()
        a, tau = self._tdl(
            batch_size=int(batch_size),
            num_time_steps=1,
            sampling_frequency=float(self.sample_rate_hz),
        )
        if hasattr(a, "numpy"):
            a = a.numpy()
        if hasattr(tau, "numpy"):
            tau = tau.numpy()
        a = self._squeeze_path_tensor(a, batch_size=batch_size, time_axis_last=True)
        tau = self._squeeze_path_tensor(tau, batch_size=batch_size, time_axis_last=False)
        if tau.shape[0] == 1 and batch_size > 1:
            tau = np.repeat(tau, batch_size, axis=0)
        exponents = np.exp(-1j * 2.0 * np.pi * tau[:, :, None] * self.frequencies_hz[None, None, :])
        h_freq = np.sum(a[:, :, None] * exponents, axis=1)
        if self.normalization:
            power = np.mean(np.abs(h_freq) ** 2, axis=1, keepdims=True)
            h_freq = h_freq / np.sqrt(np.maximum(power, 1e-12))
        return h_freq.astype(np.complex64)
