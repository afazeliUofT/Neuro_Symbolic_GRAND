from __future__ import annotations

from typing import Dict

from .fallback import FallbackTDLBackend
from .sionna_tdl import SionnaTDLBackend


def build_channel_backend(channel_cfg: Dict[str, object], profile: str, n_fft: int, seed: int, tf_threads: int = 1):
    backend_name = str(channel_cfg.get("backend", "sionna_tdl"))
    common_kwargs = dict(
        profile=str(profile),
        delay_spread_s=float(channel_cfg["delay_spread_s"]),
        subcarrier_spacing_hz=float(channel_cfg["subcarrier_spacing_hz"]),
        n_fft=int(n_fft),
        carrier_frequency_hz=float(channel_cfg["carrier_frequency_hz"]),
        min_speed_mps=float(channel_cfg.get("min_speed_mps", 0.0)),
        max_speed_mps=float(channel_cfg.get("max_speed_mps", 0.0)),
    )
    if backend_name == "sionna_tdl":
        try:
            return SionnaTDLBackend(
                normalization=bool(channel_cfg.get("normalization", True)),
                tf_intra_threads=int(tf_threads),
                tf_inter_threads=1,
                **common_kwargs,
            )
        except Exception:
            if not bool(channel_cfg.get("allow_fallback", True)):
                raise
    return FallbackTDLBackend(seed=seed, **common_kwargs)
