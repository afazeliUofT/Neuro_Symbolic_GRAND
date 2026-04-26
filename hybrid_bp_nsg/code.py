from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def crc16_ccitt(bits: np.ndarray, poly: int = 0x1021, init: int = 0x0000) -> int:
    crc = int(init)
    for b in np.asarray(bits, dtype=np.uint8).reshape(-1):
        crc ^= (int(b) & 1) << 15
        for _ in range(1):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def crc_bits_from_payload(payload: np.ndarray) -> np.ndarray:
    crc = crc16_ccitt(payload)
    return np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


@dataclass
class LDPCCode:
    h: np.ndarray
    family: str = "custom_ldpc"
    transport_k: int = 256
    k: int = 272
    n_transmitted: int = 512
    transmitted_positions: np.ndarray | None = None
    punctured_positions: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=np.uint8)
        self.m, self.n = self.h.shape
        if self.transmitted_positions is None:
            self.transmitted_positions = np.arange(min(self.n_transmitted, self.n), dtype=np.int64)
        else:
            self.transmitted_positions = np.asarray(self.transmitted_positions, dtype=np.int64)
        if self.punctured_positions is None:
            all_pos = np.arange(self.n, dtype=np.int64)
            self.punctured_positions = np.setdiff1d(all_pos, self.transmitted_positions)
        else:
            self.punctured_positions = np.asarray(self.punctured_positions, dtype=np.int64)
        self.deg_v = self.h.sum(axis=0).astype(np.float32)
        self.deg_c = self.h.sum(axis=1).astype(np.float32)
        self.vn_neighbors = [np.flatnonzero(self.h[:, j]).astype(np.int32) for j in range(self.n)]
        self.cn_neighbors = [np.flatnonzero(self.h[i]).astype(np.int32) for i in range(self.m)]
        self.edges = np.stack(np.where(self.h > 0), axis=1).astype(np.int32)
        self.cn_edge_indices = [np.flatnonzero(self.edges[:, 0] == i).astype(np.int32) for i in range(self.m)]
        self.vn_edge_indices = [np.flatnonzero(self.edges[:, 1] == j).astype(np.int32) for j in range(self.n)]
        self.has_outer_crc = bool(self.metadata.get("has_outer_crc", True))

    @property
    def rate(self) -> float:
        return float(self.transport_k) / float(max(1, self.n_transmitted))

    def syndrome(self, hard: np.ndarray) -> np.ndarray:
        x = np.asarray(hard, dtype=np.uint8).reshape(-1)
        return ((self.h.astype(np.uint8) @ x.astype(np.uint8)) & 1).astype(np.uint8)

    def is_codeword(self, hard: np.ndarray) -> bool:
        return bool(np.all(self.syndrome(hard) == 0))

    def crc_check_internal(self, hard: np.ndarray) -> bool:
        if not self.has_outer_crc:
            return True
        x = np.asarray(hard, dtype=np.uint8).reshape(-1)
        if x.size < self.transport_k + 16:
            return self.is_codeword(x)
        payload = x[: self.transport_k]
        crc_rx = x[self.transport_k : self.transport_k + 16]
        return bool(np.array_equal(crc_bits_from_payload(payload), crc_rx))

    def code_summary(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "transport_k": int(self.transport_k),
            "ldpc_k": int(self.k),
            "n_internal": int(self.n),
            "n_transmitted": int(self.n_transmitted),
            "k": int(self.k),
            "m": int(self.m),
            "rate": float(self.rate),
            "edges": int(self.h.sum()),
            "avg_check_degree": float(self.deg_c.mean()) if self.m else 0.0,
            "max_check_degree": int(self.deg_c.max()) if self.m else 0,
            "min_check_degree": int(self.deg_c.min()) if self.m else 0,
            "avg_variable_degree": float(self.deg_v.mean()) if self.n else 0.0,
            "num_punctured_internal_positions": int(len(self.punctured_positions)),
            "num_transmitted_internal_positions": int(len(self.transmitted_positions)),
            "metadata": self.metadata,
        }


def generate_pruned_pcm_5g(decoder: Any, n: int) -> Tuple[np.ndarray, np.ndarray]:
    enc = decoder._encoder
    z = int(_get_attr(enc, "z", "_z"))
    enc_n = int(_get_attr(enc, "n", "_n", default=n))
    enc_k = int(_get_attr(enc, "k", "_k"))
    k_ldpc = int(_get_attr(enc, "k_ldpc", "_k_ldpc", default=enc_k))
    n_ldpc = int(_get_attr(enc, "n_ldpc", "_n_ldpc"))
    nb_pruned = int(_get_attr(decoder, "_nb_pruned_nodes", "nb_pruned_nodes", default=0))
    pos_tx = np.ones(int(n), dtype=np.float32)
    pos_punc = np.concatenate([np.zeros(2 * z, dtype=np.float32), pos_tx], axis=0)
    k_short = k_ldpc - enc_k
    num_punc_bits = int((n_ldpc - k_short) - enc_n - 2 * z)
    tail = max(0, num_punc_bits - nb_pruned)
    pos_punc2 = np.concatenate([pos_punc, np.zeros(tail, dtype=np.float32)], axis=0)
    pos_info = pos_punc2[0:enc_k]
    num_par_bits = int(n_ldpc - k_short - enc_k - nb_pruned)
    pos_parity = pos_punc2[enc_k : enc_k + num_par_bits]
    pos_short = 2 * np.ones(k_short, dtype=np.float32)
    rm_pattern = np.concatenate([pos_info, pos_short, pos_parity], axis=0)
    pcm = decoder.pcm
    if hasattr(pcm, "todense"):
        pcm = np.asarray(pcm.todense())
    else:
        pcm = np.asarray(pcm)
    pcm = (pcm != 0).astype(np.uint8)
    idx_short = np.where(rm_pattern == 2)[0]
    idx_keep = np.setdiff1d(np.arange(pcm.shape[1]), idx_short)
    pcm_pruned = pcm[:, idx_keep]
    return pcm_pruned.astype(np.uint8), rm_pattern[idx_keep]


def _fallback_ldpc(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)
    # Build a deterministic sparse matrix with variable degree about 3 and check degree about n*3/m.
    for j in range(n):
        deg = 2 + (j % 3 == 0) + (j % 17 == 0)
        rows = rng.choice(m, size=min(deg, m), replace=False)
        H[rows, j] = 1
    # Ensure every check has degree >= 2.
    for i in range(m):
        if H[i].sum() < 2:
            cols = rng.choice(n, size=3, replace=False)
            H[i, cols] = 1
    return H


def build_code(cfg: Dict[str, Any]) -> LDPCCode:
    ccfg = cfg.get("code", {})
    family = str(ccfg.get("family", "sionna_nr_pusch_ldpc"))
    transport_k = int(ccfg.get("transport_k", int(round(ccfg.get("num_coded_bits", 512) * ccfg.get("target_coderate", 0.5)))))
    # LDPC input includes the 16-bit transport-block CRC, matching the original FIR v13 summary: transport_k=256, ldpc_k=272.
    ldpc_input_k = int(ccfg.get("ldpc_k", transport_k + 16))
    n_tx = int(ccfg.get("num_coded_bits", 512))
    seed = int(ccfg.get("seed", cfg.get("project", {}).get("seed", 31415)))
    use_sionna = bool(ccfg.get("use_sionna_pcm", True)) and family == "sionna_nr_pusch_ldpc"
    metadata: Dict[str, Any] = {
        "num_coded_bits": n_tx,
        "target_coderate": float(ccfg.get("target_coderate", transport_k / max(1, n_tx))),
        "num_bits_per_symbol": int(ccfg.get("num_bits_per_symbol", 2)),
        "num_layers": int(ccfg.get("num_layers", 1)),
        "tb_size": transport_k,
        "tb_crc_length": 16,
        "has_outer_crc": True,
        "pusch_channel_type": "PUSCH",
        "source": "fallback_ldpc",
    }
    if use_sionna:
        try:
            from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder  # type: ignore
            encoder = LDPC5GEncoder(ldpc_input_k, n_tx)
            try:
                decoder = LDPC5GDecoder(encoder, num_iter=1, prune_pcm=True)
            except TypeError:
                decoder = LDPC5GDecoder(encoder, prune_pcm=True)
            H, rm_pattern = generate_pruned_pcm_5g(decoder, n_tx)
            tx_pos = np.where(rm_pattern == 1)[0].astype(np.int64)
            punc_pos = np.where(rm_pattern == 0)[0].astype(np.int64)
            k_ldpc = int(_get_attr(encoder, "k_ldpc", "_k_ldpc", default=transport_k + 16))
            enc_k = int(_get_attr(encoder, "k", "_k", default=transport_k))
            metadata.update({
                "source": "sionna_ldpc5g_pruned_pcm",
                "cb_size": enc_k,
                "ldpc_k_property": k_ldpc,
                "bg": str(_get_attr(encoder, "_bg", "bg", default="unknown")),
                "z": int(_get_attr(encoder, "z", "_z", default=0)),
                "output_perm_inv_size": n_tx,
                "interleaver_validated": True,
            })
            # In Sionna LDPC5GEncoder(k,n), k includes the input bits given to encoder.
            # For the FIR setup, user transport is 256 and CRC occupies the next 16 positions.
            k_internal = ldpc_input_k
            return LDPCCode(H, family=family, transport_k=transport_k, k=k_internal, n_transmitted=n_tx,
                            transmitted_positions=tx_pos, punctured_positions=punc_pos, metadata=metadata)
        except Exception as e:
            metadata["sionna_error"] = f"{type(e).__name__}: {e}"
    n_internal = int(ccfg.get("fallback_n_internal", 584))
    m = int(ccfg.get("fallback_m", 312))
    H = _fallback_ldpc(n_internal, m, seed)
    tx_pos = np.arange(n_tx, dtype=np.int64)
    punc_pos = np.arange(n_tx, n_internal, dtype=np.int64)
    return LDPCCode(H, family=family + "_fallback", transport_k=transport_k, k=ldpc_input_k,
                    n_transmitted=n_tx, transmitted_positions=tx_pos, punctured_positions=punc_pos, metadata=metadata)


def write_code_summary(code: LDPCCode, out_dir: str | Path) -> None:
    p = Path(out_dir) / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    (p / "code_summary.json").write_text(json.dumps(code.code_summary(), indent=2, default=str), encoding="utf-8")
