from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .peg_ldpc import LDPCCode
from .sionna_nr_ldpc import build_sionna_nr_ldpc


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
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def _scalar_int(x: Any) -> int:
    arr = _to_numpy(x).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty scalar conversion")
    return int(arr[0])


def _import_nr_blocks():
    tried = []
    try:
        from sionna.phy.nr import TBEncoder, TBDecoder  # type: ignore
        from sionna.phy.nr.utils import calculate_tb_size  # type: ignore
        return TBEncoder, TBDecoder, calculate_tb_size
    except Exception as e:
        tried.append(f"sionna.phy.nr: {e}")
    try:
        from sionna.nr import TBEncoder, TBDecoder  # type: ignore
        from sionna.nr.utils import calculate_tb_size  # type: ignore
        return TBEncoder, TBDecoder, calculate_tb_size
    except Exception as e:
        tried.append(f"sionna.nr: {e}")
    raise ImportError("Could not import Sionna NR TBEncoder/TBDecoder/calculate_tb_size. Tried: " + " | ".join(tried))


def _call_block(block, x):
    try:
        y = block(x)
    except Exception:
        y = block(np.asarray(x, dtype=np.float32))
    return _to_numpy(y)


def _append_tb_crc(tb_enc, payload_bits: np.ndarray, cb_size: int) -> np.ndarray:
    """Attach the outer TB CRC exactly as Sionna TBEncoder does.

    This function is deliberately strict: if the exposed CRC encoder API is not usable,
    we raise instead of silently returning the raw payload. Silent fallback would create
    parity-valid but CRC-invalid code-block inputs and poison both training labels and
    rescue acceptance.
    """
    arr = np.asarray(payload_bits, dtype=np.uint8)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    work = arr.astype(np.float32)
    tb_crc_enc = getattr(tb_enc, "tb_crc_encoder", None)
    if tb_crc_enc is None:
        raise RuntimeError("TBEncoder does not expose tb_crc_encoder; cannot build CRC-aware CB input.")
    try:
        work = _call_block(tb_crc_enc, work)
    except Exception as e:
        raise RuntimeError(f"Failed to apply Sionna TB CRC encoder: {e}") from e
    work = (_to_numpy(work) > 0.5).astype(np.uint8)
    k_padding = int(getattr(tb_enc, "k_padding", 0) or 0)
    if k_padding > 0:
        pad = np.zeros((work.shape[0], k_padding), dtype=np.uint8)
        work = np.concatenate([work, pad], axis=1)
    if work.shape[1] < int(cb_size):
        pad = np.zeros((work.shape[0], int(cb_size) - work.shape[1]), dtype=np.uint8)
        work = np.concatenate([work, pad], axis=1)
    if work.shape[1] > int(cb_size):
        work = work[:, : int(cb_size)]
    return work[0] if squeeze else work


def build_sionna_nr_pusch_ldpc(
    num_coded_bits: int = 512,
    target_coderate: float = 0.5,
    num_bits_per_symbol: int = 2,
    num_layers: int = 1,
    target_tb_size: int | None = None,
    use_scrambler: bool = False,
    n_rnti: int = 1,
    n_id: int = 1,
    strict_pcm_check: bool = True,
    seed: int = 1234,
    **_,
) -> LDPCCode:
    """Build a short-block 5G NR PUSCH-derived LDPC code object.

    The package keeps the rescue algorithm in the LDPC internal domain, but the code/block
    sizing, TB CRC, code-block sizing, and output interleaver are derived from Sionna NR's
    PUSCH transport-block tools. For simplicity and codeword-consistent rescue, scrambling is
    disabled by default. This keeps the output-bit interleaver standard while making the
    internal<->transmitted mapping explicit and verifiable.
    """
    TBEncoder, TBDecoder, calculate_tb_size = _import_nr_blocks()

    if bool(use_scrambler):
        raise NotImplementedError(
            "This package keeps the PUSCH transport-block CRC/interleaver path but requires use_scrambler=False so that the internal<->transmitted mapping remains explicit for rescue search."
        )

    calc_out = calculate_tb_size(
        modulation_order=int(num_bits_per_symbol),
        target_coderate=float(target_coderate),
        target_tb_size=target_tb_size,
        num_coded_bits=int(num_coded_bits),
        num_layers=int(num_layers),
        return_cw_length=True,
        verbose=False,
    )
    if len(calc_out) != 6:
        raise RuntimeError(f"Unexpected calculate_tb_size() output length {len(calc_out)}")
    tb_size, cb_size, num_cb, tb_crc_len, cb_crc_len, cw_length = calc_out
    tb_size = _scalar_int(tb_size)
    cb_size = _scalar_int(cb_size)
    num_cb = _scalar_int(num_cb)
    tb_crc_len = _scalar_int(tb_crc_len)
    cb_crc_len = _scalar_int(cb_crc_len)
    cw_length = _to_numpy(cw_length).reshape(-1).astype(np.int64)
    cw_length = cw_length[cw_length > 0]
    if num_cb != 1 or cw_length.size != 1:
        raise RuntimeError(
            "This PUSCH rescue package currently supports a single code block only. "
            f"Sionna returned num_cb={num_cb}, cw_length={cw_length.tolist()} for num_coded_bits={num_coded_bits}."
        )
    rm_n = int(cw_length[0])

    tb_enc = TBEncoder(
        target_tb_size=int(tb_size),
        num_coded_bits=int(num_coded_bits),
        target_coderate=float(target_coderate),
        num_bits_per_symbol=int(num_bits_per_symbol),
        num_layers=int(num_layers),
        n_rnti=int(n_rnti),
        n_id=int(n_id),
        channel_type="PUSCH",
        codeword_index=0,
        use_scrambler=bool(use_scrambler),
        verbose=False,
    )
    tb_dec = TBDecoder(tb_enc, num_bp_iter=20)

    # Underlying 5G LDPC code block used by the BP/GRAND rescue path.
    ldpc = build_sionna_nr_ldpc(k=int(cb_size), n=rm_n, strict_pcm_check=bool(strict_pcm_check), seed=int(seed))

    output_perm_inv = _to_numpy(getattr(tb_enc, "output_perm_inv")).reshape(-1).astype(np.int64)
    if output_perm_inv.size != int(num_coded_bits):
        output_perm_inv = output_perm_inv[: int(num_coded_bits)]
    perm_from_inv = np.argsort(output_perm_inv)

    def interleave_rm_bits(bits_rm: np.ndarray, use_alt: bool = False) -> np.ndarray:
        arr = np.asarray(bits_rm, dtype=np.uint8)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        if arr.shape[1] != int(num_coded_bits):
            raise ValueError(f"Expected {num_coded_bits} rate-matched bits, got {arr.shape[1]}")
        if use_alt:
            out = arr[:, output_perm_inv]
        else:
            out = arr[:, perm_from_inv]
        return out[0] if squeeze else out

    def deinterleave_tx_bits(bits_tx: np.ndarray, use_alt: bool = False) -> np.ndarray:
        arr = np.asarray(bits_tx, dtype=np.uint8)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        if arr.shape[1] != int(num_coded_bits):
            raise ValueError(f"Expected {num_coded_bits} transmitted bits, got {arr.shape[1]}")
        if use_alt:
            out = arr[:, perm_from_inv]
        else:
            out = arr[:, output_perm_inv]
        return out[0] if squeeze else out

    def tb_encode_bits(payload_bits: np.ndarray) -> np.ndarray:
        arr = np.asarray(payload_bits, dtype=np.uint8)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        out = _call_block(tb_enc, arr.astype(np.float32))
        out = (out > 0.5).astype(np.uint8)
        return out[0] if squeeze else out

    def cb_input_from_payload(payload_bits: np.ndarray) -> np.ndarray:
        return _append_tb_crc(tb_enc, payload_bits, int(cb_size))

    # Validate the output interleaver orientation against Sionna's TB encoder.
    chosen_alt = None
    rng = np.random.default_rng(int(seed) + 17)
    if not bool(use_scrambler):
        for alt in (False, True):
            ok = True
            for _ in range(4):
                payload = rng.integers(0, 2, size=tb_size, dtype=np.uint8)
                cb_in = cb_input_from_payload(payload)
                rm_bits = ldpc.encode(cb_in)
                tx_bits = interleave_rm_bits(rm_bits, use_alt=alt)
                sionna_bits = tb_encode_bits(payload)
                if not np.array_equal(tx_bits.reshape(-1), sionna_bits.reshape(-1)):
                    ok = False
                    break
            if ok:
                chosen_alt = alt
                break
        if chosen_alt is None:
            raise RuntimeError(
                "Could not validate PUSCH output-bit interleaver mapping against Sionna TBEncoder. "
                "This usually means the local Sionna build exposes a different permutation semantic."
            )

    def rm_to_tx(bits_rm: np.ndarray) -> np.ndarray:
        return interleave_rm_bits(bits_rm, use_alt=bool(chosen_alt))

    def tx_to_rm_bits(bits_tx: np.ndarray) -> np.ndarray:
        return deinterleave_tx_bits(bits_tx, use_alt=bool(chosen_alt))

    def tx_llr_to_internal(llr_tx: np.ndarray, info_llr: np.ndarray | None = None) -> np.ndarray:
        arr = np.asarray(llr_tx, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        rm_llr = arr[:, output_perm_inv] if not bool(chosen_alt) else arr[:, perm_from_inv]
        internal = ldpc.expand_llr(rm_llr, info_llr=info_llr)
        return internal[0] if squeeze else internal

    def internal_to_tx(bits_internal: np.ndarray) -> np.ndarray:
        rm = ldpc.internal_to_tx(bits_internal)
        return rm_to_tx(rm)

    def encode_payload(payload_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cb_in = cb_input_from_payload(payload_bits)
        internal = ldpc.encode_internal(cb_in)
        tx = internal_to_tx(internal)
        return internal, tx

    def payload_from_internal(bits_internal: np.ndarray) -> np.ndarray:
        arr = np.asarray(bits_internal, dtype=np.uint8)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        out = arr[:, : tb_size]
        return out[0] if squeeze else out

    def crc_check_internal(bits_internal: np.ndarray) -> bool:
        """Check the outer TB CRC directly in the internal systematic domain.

        For the single-code-block setting used by this package, the first `cb_size` internal
        bits equal the code-block input to the 5G LDPC encoder: payload bits, TB CRC bits,
        and optional zero padding. Comparing these bits against a freshly regenerated
        Sionna-consistent TB-CRC attachment avoids any ambiguity about decoder logit sign
        conventions while remaining exactly aligned with the transport-block definition.
        """
        arr = np.asarray(bits_internal, dtype=np.uint8)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        if arr.shape[1] != ldpc.n:
            raise ValueError(f"Expected internal length {ldpc.n}, got {arr.shape[1]}")
        payload = payload_from_internal(arr)
        expected_cb = cb_input_from_payload(payload)
        actual_cb = arr[:, : int(cb_size)].astype(np.uint8)
        ok = np.all(actual_cb == expected_cb.astype(np.uint8), axis=1)
        return bool(ok[0]) if squeeze else ok

    code = LDPCCode(
        h=ldpc.h,
        k=ldpc.k,
        n=ldpc.n,
        family="sionna_nr_pusch_ldpc",
        rate=float(tb_size) / float(num_coded_bits),
        g=None,
        encoder=ldpc.encoder,
        rm_pattern=ldpc.rm_pattern,
        bg=ldpc.bg,
        metadata={
            "num_coded_bits": int(num_coded_bits),
            "target_coderate": float(target_coderate),
            "num_bits_per_symbol": int(num_bits_per_symbol),
            "num_layers": int(num_layers),
            "tb_size": int(tb_size),
            "cb_size": int(cb_size),
            "num_cb": int(num_cb),
            "tb_crc_length": int(tb_crc_len),
            "cb_crc_length": int(cb_crc_len),
            "cw_length": [int(x) for x in cw_length.tolist()],
            "use_scrambler": bool(use_scrambler),
            "interleaver_validated": not bool(use_scrambler),
            "output_perm_inv_size": int(output_perm_inv.size),
            "has_outer_crc": bool(tb_crc_len > 0 or cb_crc_len > 0),
            "pusch_channel_type": "PUSCH",
        },
        payload_k=int(tb_size),
        encode_payload_fn=encode_payload,
        internal_to_tx_fn=internal_to_tx,
        tx_llr_to_internal_fn=tx_llr_to_internal,
        crc_check_internal_fn=crc_check_internal,
        payload_from_internal_fn=payload_from_internal,
    )
    code.tb_encoder = tb_enc
    code.tb_decoder = tb_dec
    code.output_perm_inv = output_perm_inv
    code.output_perm = perm_from_inv if not bool(chosen_alt) else output_perm_inv
    code.cb_input_from_payload = cb_input_from_payload
    code.rm_to_tx_bits = rm_to_tx
    code.tx_to_rm_bits = tx_to_rm_bits
    code.underlying_ldpc = ldpc
    return code
