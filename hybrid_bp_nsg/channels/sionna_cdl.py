from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .qpsk import bits_to_qpsk, qpsk_to_llr


@dataclass(frozen=True)
class CDLConfig:
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    num_ofdm_symbols: int = 14
    fft_size: int = 72
    cyclic_prefix_length: int = 0
    delay_spread_s: float = 100e-9
    speed_m_per_s: float = 0.0
    normalize_channel: bool = True
    direction: str = "uplink"
    model: str = "C"
    perfect_csi: bool = True


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import tensorflow as tf  # type: ignore
        if isinstance(x, tf.Tensor):
            x = x.numpy()
    except Exception:
        pass
    if hasattr(x, "numpy"):
        try:
            x = x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def _import_sionna_blocks():
    tried = []
    ResourceGrid = None
    OFDMChannel = None
    CDL = None
    AntennaArray = None
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(f"TensorFlow import failed for Sionna CDL channel: {e}") from e

    for mod in ["sionna.phy.ofdm", "sionna.ofdm"]:
        try:
            m = __import__(mod, fromlist=["ResourceGrid"])
            ResourceGrid = getattr(m, "ResourceGrid")
            break
        except Exception as e:
            tried.append(f"{mod}.ResourceGrid: {e}")
    for mod in ["sionna.phy.channel", "sionna.channel"]:
        try:
            m = __import__(mod, fromlist=["OFDMChannel"])
            OFDMChannel = getattr(m, "OFDMChannel")
            break
        except Exception as e:
            tried.append(f"{mod}.OFDMChannel: {e}")
    for mod in ["sionna.phy.channel.tr38901", "sionna.channel.tr38901"]:
        try:
            m = __import__(mod, fromlist=["CDL", "AntennaArray"])
            CDL = getattr(m, "CDL")
            AntennaArray = getattr(m, "AntennaArray")
            break
        except Exception as e:
            tried.append(f"{mod}.CDL/AntennaArray: {e}")
    if ResourceGrid is None or OFDMChannel is None or CDL is None or AntennaArray is None:
        raise ImportError("Could not import Sionna OFDM/CDL blocks. Tried: " + " | ".join(tried))
    return tf, ResourceGrid, OFDMChannel, CDL, AntennaArray


def _make_antenna_array(AntennaArray, carrier_frequency_hz: float):
    tries = [
        dict(
            num_rows=1,
            num_cols=1,
            polarization="single",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency_hz,
            horizontal_spacing=0.5,
            vertical_spacing=0.5,
        ),
        dict(
            num_rows=1,
            num_cols=1,
            polarization="single",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency_hz,
            horizontal_spacing=0.5,
            vertical_spacing=0.5,
        ),
        dict(
            num_rows=1,
            num_cols=1,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency_hz,
            horizontal_spacing=0.5,
            vertical_spacing=0.5,
        ),
    ]
    last = None
    for kw in tries:
        try:
            return AntennaArray(**kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not construct Sionna AntennaArray: {last}")


def _make_resource_grid(ResourceGrid, cfg: CDLConfig):
    tries = [
        dict(
            num_ofdm_symbols=int(cfg.num_ofdm_symbols),
            fft_size=int(cfg.fft_size),
            subcarrier_spacing=float(cfg.subcarrier_spacing_hz),
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=int(cfg.cyclic_prefix_length),
            num_guard_carriers=(0, 0),
            dc_null=False,
            pilot_pattern=None,
            pilot_ofdm_symbol_indices=None,
        ),
        dict(
            num_ofdm_symbols=int(cfg.num_ofdm_symbols),
            fft_size=int(cfg.fft_size),
            subcarrier_spacing=float(cfg.subcarrier_spacing_hz),
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=int(cfg.cyclic_prefix_length),
            num_guard_carriers=(0, 0),
            dc_null=False,
            pilot_pattern="empty",
            pilot_ofdm_symbol_indices=None,
        ),
    ]
    last = None
    for kw in tries:
        try:
            return ResourceGrid(**kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not construct Sionna ResourceGrid: {last}")


class CDLLink:
    def __init__(self, cfg: CDLConfig):
        self.cfg = cfg
        self.tf, ResourceGrid, OFDMChannel, CDL, AntennaArray = _import_sionna_blocks()
        self.resource_grid = _make_resource_grid(ResourceGrid, cfg)
        ut_array = _make_antenna_array(AntennaArray, float(cfg.carrier_frequency_hz))
        bs_array = _make_antenna_array(AntennaArray, float(cfg.carrier_frequency_hz))

        cdl_kwargs = dict(
            model=str(cfg.model),
            delay_spread=float(cfg.delay_spread_s),
            carrier_frequency=float(cfg.carrier_frequency_hz),
            ut_array=ut_array,
            bs_array=bs_array,
            direction=str(cfg.direction),
        )
        # Older TensorFlow-based Sionna often supports min_speed; some versions support max_speed too.
        try:
            self.channel_model = CDL(min_speed=float(cfg.speed_m_per_s), max_speed=float(cfg.speed_m_per_s), **cdl_kwargs)
        except Exception:
            self.channel_model = CDL(min_speed=float(cfg.speed_m_per_s), **cdl_kwargs)
        self.channel = OFDMChannel(
            self.channel_model,
            self.resource_grid,
            normalize_channel=bool(cfg.normalize_channel),
            return_channel=True,
        )
        self.device = "/GPU:0" if self.tf.config.list_physical_devices("GPU") else "/CPU:0"
        self.num_re = int(cfg.num_ofdm_symbols) * int(cfg.fft_size)

    def _call_ofdm_channel(self, x_tf, no_tf):
        """Compatibility wrapper for Sionna OFDMChannel call signatures.

        Sionna releases differ here:
          * TensorFlow-era releases may expect ``channel(x, no)``
          * older examples use ``channel([x, no])``
          * some builds also accept ``channel((x, no))``
        We try the current preferred signature first and fall back only if needed.
        """
        last = None
        for style in ("separate", "tuple", "list"):
            try:
                if style == "separate":
                    return self.channel(x_tf, no_tf)
                if style == "tuple":
                    return self.channel((x_tf, no_tf))
                return self.channel([x_tf, no_tf])
            except Exception as e:
                last = e
        raise last

    def simulate_llr(self, tx_bits: np.ndarray, noise_var: float, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, object]]:
        if seed is not None:
            try:
                self.tf.random.set_seed(int(seed))
            except Exception:
                pass
        tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
        if tx_bits.size % 2 != 0:
            raise ValueError("QPSK requires an even number of transmitted bits")
        syms = bits_to_qpsk(tx_bits)
        num_syms = syms.size
        if num_syms > self.num_re:
            raise ValueError(f"Need {num_syms} RE for QPSK symbols but resource grid has only {self.num_re}")
        x_rg = np.zeros((1, 1, 1, int(self.cfg.num_ofdm_symbols), int(self.cfg.fft_size)), dtype=np.complex64)
        flat = x_rg.reshape(1, 1, 1, -1)
        flat[..., :num_syms] = syms[None, None, None, :]
        x_rg = flat.reshape(x_rg.shape)
        with self.tf.device(self.device):
            x_tf = self.tf.convert_to_tensor(x_rg, dtype=self.tf.complex64)
            # A scalar float32 noise power is accepted by current Sionna builds and
            # keeps broadcasting simple across TF-era releases.
            no_tf = self.tf.constant(float(noise_var), dtype=self.tf.float32)
            y, h = self._call_ofdm_channel(x_tf, no_tf)
        y_np = _to_numpy(y)
        h_np = _to_numpy(h)
        y_arr = np.asarray(y_np)
        h_arr = np.asarray(h_np)
        # Collapse leading batch/antenna dimensions for the 1x1 case.
        y_arr = np.squeeze(y_arr)
        h_arr = np.squeeze(h_arr)
        while h_arr.ndim > 2:
            h_arr = h_arr[0]
        while y_arr.ndim > 2:
            y_arr = y_arr[0]
        y_used = y_arr.reshape(-1)[:num_syms].astype(np.complex64)
        h_used = h_arr.reshape(-1)[:num_syms].astype(np.complex64)
        llr = qpsk_to_llr(y_used, float(noise_var), h=h_used if bool(self.cfg.perfect_csi) else None)
        meta = {
            "channel_type": f"SIONNA_CDL_{self.cfg.model}",
            "perfect_csi": bool(self.cfg.perfect_csi),
            "equalizer": "one_tap_perfect_csi",
            "num_re": int(self.num_re),
            "num_data_re": int(num_syms),
            "device": self.device,
        }
        return llr.astype(np.float32), meta


_LINK_CACHE: Dict[CDLConfig, CDLLink] = {}


def simulate_cdl_c_qpsk(tx_bits: np.ndarray, noise_var: float, seed: int | None = None, **cfg_kwargs) -> Tuple[np.ndarray, Dict[str, object]]:
    cfg = CDLConfig(**cfg_kwargs)
    link = _LINK_CACHE.get(cfg)
    if link is None:
        link = CDLLink(cfg)
        _LINK_CACHE[cfg] = link
    return link.simulate_llr(tx_bits=np.asarray(tx_bits, dtype=np.uint8), noise_var=float(noise_var), seed=seed)
