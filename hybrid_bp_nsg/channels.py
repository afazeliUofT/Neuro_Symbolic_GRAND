from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    # Treat config SNR as Eb/N0. For QPSK, Es/N0 = Eb/N0 * rate * bits/symbol.
    ebno = 10 ** (float(snr_db) / 10.0)
    esno = max(1e-12, ebno * max(rate, 1e-6) * max(bits_per_symbol, 1))
    return 1.0 / esno


def _bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if arr.size % 2:
        raise ValueError(f"QPSK requires an even number of bits, got {arr.size}")
    pairs = arr.reshape(-1, 2).astype(np.float32)
    syms = ((1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])) / math.sqrt(2.0)
    return syms.astype(np.complex64)


def _qpsk_to_llr(y: np.ndarray, noise_var: float, h: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.complex64).reshape(-1)
    if h is None:
        r = y
    else:
        h = np.asarray(h, dtype=np.complex64).reshape(-1)
        if h.size != y.size:
            raise ValueError(f"Expected CSI length {y.size}, got {h.size}")
        r = np.conj(h) * y
    scale = float(2.0 * math.sqrt(2.0) / max(float(noise_var), 1e-12))
    out = np.empty(2 * y.size, dtype=np.float32)
    out[0::2] = (scale * np.real(r)).astype(np.float32)
    out[1::2] = (scale * np.imag(r)).astype(np.float32)
    return out


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import tensorflow as tf  # type: ignore
        if isinstance(x, tf.Tensor):
            return x.numpy()
    except Exception:
        pass
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)


def _import_sionna_blocks():
    tried = []
    ResourceGrid = OFDMChannel = CDL = AntennaArray = None
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
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
        dict(num_rows=1, num_cols=1, polarization="single", antenna_pattern="omni", carrier_frequency=carrier_frequency_hz, horizontal_spacing=0.5, vertical_spacing=0.5),
        dict(num_rows=1, num_cols=1, polarization="single", antenna_pattern="38.901", carrier_frequency=carrier_frequency_hz, horizontal_spacing=0.5, vertical_spacing=0.5),
        dict(num_rows=1, num_cols=1, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_frequency_hz, horizontal_spacing=0.5, vertical_spacing=0.5),
    ]
    last = None
    for kw in tries:
        try:
            return AntennaArray(**kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not construct Sionna AntennaArray: {last}")


def _make_resource_grid(ResourceGrid, cfg: Dict[str, Any]):
    num_ofdm_symbols = int(cfg.get("num_ofdm_symbols", 14))
    fft_size = int(cfg.get("fft_size", 72))
    subcarrier_spacing_hz = float(cfg.get("subcarrier_spacing_hz", 30e3))
    cyclic_prefix_length = int(cfg.get("cyclic_prefix_length", 0))
    tries = [
        dict(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size, subcarrier_spacing=subcarrier_spacing_hz, num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=cyclic_prefix_length, num_guard_carriers=(0, 0), dc_null=False, pilot_pattern=None, pilot_ofdm_symbol_indices=None),
        dict(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size, subcarrier_spacing=subcarrier_spacing_hz, num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=cyclic_prefix_length, num_guard_carriers=(0, 0), dc_null=False, pilot_pattern="empty", pilot_ofdm_symbol_indices=None),
    ]
    last = None
    for kw in tries:
        try:
            return ResourceGrid(**kw)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not construct Sionna ResourceGrid: {last}")


class _SionnaCDLLink:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.tf, ResourceGrid, OFDMChannel, CDL, AntennaArray = _import_sionna_blocks()
        self.resource_grid = _make_resource_grid(ResourceGrid, self.cfg)
        self.carrier_frequency_hz = float(self.cfg.get("carrier_frequency_hz", 3.5e9))
        self.delay_spread_s = float(self.cfg.get("delay_spread_s", 100e-9))
        self.speed_m_per_s = float(self.cfg.get("speed_m_per_s", 0.0))
        self.normalize_channel = bool(self.cfg.get("normalize_channel", True))
        self.direction = str(self.cfg.get("direction", "uplink"))
        self.model = str(self.cfg.get("model", "C"))
        self.perfect_csi = bool(self.cfg.get("perfect_csi", True))
        self.num_ofdm_symbols = int(self.cfg.get("num_ofdm_symbols", 14))
        self.fft_size = int(self.cfg.get("fft_size", 72))
        ut_array = _make_antenna_array(AntennaArray, self.carrier_frequency_hz)
        bs_array = _make_antenna_array(AntennaArray, self.carrier_frequency_hz)
        cdl_kwargs = dict(
            model=self.model,
            delay_spread=self.delay_spread_s,
            carrier_frequency=self.carrier_frequency_hz,
            ut_array=ut_array,
            bs_array=bs_array,
            direction=self.direction,
        )
        try:
            self.channel_model = CDL(min_speed=self.speed_m_per_s, max_speed=self.speed_m_per_s, **cdl_kwargs)
        except Exception:
            self.channel_model = CDL(min_speed=self.speed_m_per_s, **cdl_kwargs)
        self.channel = OFDMChannel(self.channel_model, self.resource_grid, normalize_channel=self.normalize_channel, return_channel=True)
        self.device = "/GPU:0" if self.tf.config.list_physical_devices("GPU") else "/CPU:0"
        self.num_re = self.num_ofdm_symbols * self.fft_size

    def _call_channel(self, x_tf, no_tf):
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
        if tx_bits.size % 2:
            raise ValueError("QPSK requires an even number of transmitted bits")
        syms = _bits_to_qpsk(tx_bits)
        if syms.size > self.num_re:
            raise ValueError(f"Need {syms.size} RE for QPSK symbols but resource grid has only {self.num_re}")
        x_rg = np.zeros((1, 1, 1, self.num_ofdm_symbols, self.fft_size), dtype=np.complex64)
        flat = x_rg.reshape(1, 1, 1, -1)
        flat[..., : syms.size] = syms[None, None, None, :]
        x_rg = flat.reshape(x_rg.shape)
        with self.tf.device(self.device):
            x_tf = self.tf.convert_to_tensor(x_rg, dtype=self.tf.complex64)
            no_tf = self.tf.constant(float(noise_var), dtype=self.tf.float32)
            y, h = self._call_channel(x_tf, no_tf)
        y_arr = np.squeeze(np.asarray(_to_numpy(y)))
        h_arr = np.squeeze(np.asarray(_to_numpy(h)))
        while y_arr.ndim > 2:
            y_arr = y_arr[0]
        while h_arr.ndim > 2:
            h_arr = h_arr[0]
        y_used = y_arr.reshape(-1)[: syms.size].astype(np.complex64)
        h_used = h_arr.reshape(-1)[: syms.size].astype(np.complex64)
        llr = _qpsk_to_llr(y_used, float(noise_var), h=h_used if self.perfect_csi else None)
        meta = {
            "channel_type": f"SIONNA_CDL_{self.model}",
            "perfect_csi": self.perfect_csi,
            "equalizer": "one_tap_perfect_csi",
            "num_re": int(self.num_re),
            "num_data_re": int(syms.size),
            "device": self.device,
        }
        return llr.astype(np.float32), meta


_CDL_CACHE: Dict[tuple, _SionnaCDLLink] = {}


def _simulate_sionna_cdl(tx_bits: np.ndarray, noise_var: float, rng: np.random.Generator, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, object]]:
    key = tuple(sorted((str(k), repr(v)) for k, v in dict(cfg or {}).items()))
    link = _CDL_CACHE.get(key)
    if link is None:
        link = _SionnaCDLLink(dict(cfg or {}))
        _CDL_CACHE[key] = link
    return link.simulate_llr(tx_bits, noise_var, seed=int(rng.integers(0, 2**31 - 1)))


def _simulate_cdl_surrogate(tx_bits: np.ndarray, no: float, rng: np.random.Generator, n_tx: int) -> Tuple[np.ndarray, str, str]:
    if n_tx % 2:
        tx_bits2 = np.concatenate([tx_bits, np.zeros(1, dtype=np.uint8)])
    else:
        tx_bits2 = tx_bits
    b0 = tx_bits2[0::2].astype(np.float32)
    b1 = tx_bits2[1::2].astype(np.float32)
    x = ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) / math.sqrt(2.0)
    n_sym = x.size
    block = max(2, int(math.sqrt(max(1, n_sym))))
    h_blocks = (rng.normal(size=(n_sym + block - 1) // block) + 1j * rng.normal(size=(n_sym + block - 1) // block)) / math.sqrt(2.0)
    h = np.repeat(h_blocks, block)[:n_sym]
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
    return out[:n_tx].astype(np.float32), "SIONNA_CDL_C_SURROGATE", "one_tap_perfect_csi"


def simulate_frame(code: LDPCCode, snr_db: float, profile: str, rng: np.random.Generator, cfg: Dict) -> Frame:
    n_tx = int(code.n_transmitted)
    cw_internal = np.zeros(code.n, dtype=np.uint8)
    tx_bits = cw_internal[code.transmitted_positions]
    no = _noise_var_from_snr(snr_db, rate=code.rate, bits_per_symbol=int(cfg.get("code", {}).get("num_bits_per_symbol", 2)))
    profile_u = str(profile).upper().replace("-", "_")

    if profile_u in {"AWGN", "A"}:
        syms = _bits_to_qpsk(tx_bits)
        sigma = math.sqrt(max(no, 1e-12) / 2.0)
        noise = (sigma * rng.normal(size=syms.shape) + 1j * sigma * rng.normal(size=syms.shape)).astype(np.complex64)
        y = syms + noise
        llr_tx = _qpsk_to_llr(y, no).astype(np.float32)
        channel_type = "AWGN"
        equalizer = "identity"
    elif profile_u in {"CDL_C", "CDLC", "C"}:
        cdl_cfg = dict(cfg.get("channel", {}).get("cdl_c", {}))
        if bool(cdl_cfg.get("use_sionna", True)):
            try:
                llr_tx, meta = _simulate_sionna_cdl(tx_bits, no, rng, cdl_cfg)
                channel_type = str(meta.get("channel_type", "SIONNA_CDL_C"))
                equalizer = str(meta.get("equalizer", "one_tap_perfect_csi"))
            except Exception as e:
                if bool(cdl_cfg.get("strict_sionna", False)):
                    raise
                llr_tx, channel_type, equalizer = _simulate_cdl_surrogate(tx_bits, no, rng, n_tx)
                channel_type = f"SIONNA_CDL_C_SURROGATE_FALLBACK[{type(e).__name__}]"
        else:
            llr_tx, channel_type, equalizer = _simulate_cdl_surrogate(tx_bits, no, rng, n_tx)
    else:
        raise ValueError(f"Unknown channel profile {profile!r}; supported profiles: AWGN, CDL_C")

    llr_internal = np.zeros(code.n, dtype=np.float32)
    llr_internal[code.transmitted_positions] = llr_tx[: code.transmitted_positions.size]
    if code.punctured_positions.size:
        llr_internal[code.punctured_positions] = 0.0
    llr_internal = np.clip(llr_internal, -80.0, 80.0).astype(np.float32)
    return Frame(
        llr_internal=llr_internal,
        tx_bits=tx_bits.copy(),
        codeword_internal=cw_internal,
        profile=str(profile),
        snr_db=float(snr_db),
        channel_type=channel_type,
        equalizer=equalizer,
    )


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
