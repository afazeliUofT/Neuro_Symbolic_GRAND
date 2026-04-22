from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .peg_ldpc import LDPCCode


def _import_encoder_decoder():
    tried = []
    enc = dec = None
    paths = [
        ("sionna.phy.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("sionna.fec.ldpc.encoding", "LDPC5GEncoder"),
        ("sionna.fec.ldpc", "LDPC5GEncoder"),
    ]
    for mod, name in paths:
        try:
            m = __import__(mod, fromlist=[name])
            enc = getattr(m, name)
            break
        except Exception as e:
            tried.append(f"{mod}.{name}: {e}")
    paths = [
        ("sionna.phy.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("sionna.fec.ldpc.decoding", "LDPC5GDecoder"),
        ("sionna.fec.ldpc", "LDPC5GDecoder"),
    ]
    for mod, name in paths:
        try:
            m = __import__(mod, fromlist=[name])
            dec = getattr(m, name)
            break
        except Exception as e:
            tried.append(f"{mod}.{name}: {e}")
    if enc is None:
        raise ImportError("Could not import Sionna 5G LDPC encoder. Tried: " + " | ".join(tried))
    return enc, dec


def _to_numpy(x: Any) -> np.ndarray:
    """Convert TensorFlow/PyTorch/SciPy-like objects to NumPy without importing heavy stacks eagerly."""
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
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


def _get_attr(obj: Any, names: list[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _instantiate_encoder(k: int, n: int):
    Enc, _ = _import_encoder_decoder()
    # Sionna signatures differ slightly across releases. Try the modern/simple calls first.
    errors = []
    for kwargs in ({}, {"num_bits_per_symbol": None}):
        try:
            return Enc(k, n, **kwargs)
        except TypeError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(f"Could not instantiate LDPC5GEncoder(k={k}, n={n}). Errors: {errors}")


def _extract_full_pcm(enc) -> np.ndarray:
    candidates = ["pcm", "_pcm", "pcm_dense", "_pcm_dense", "_pcm_csr", "pcm_csr"]
    for name in candidates:
        if hasattr(enc, name):
            pcm = _to_numpy(getattr(enc, name))
            if pcm.ndim == 2 and pcm.size:
                return (pcm.astype(np.uint8) & 1)
    # Some Sionna versions keep the matrix on the decoder's encoder object.
    try:
        _, Dec = _import_encoder_decoder()
        if Dec is not None:
            dec = Dec(enc, num_iter=1)
            for obj in [dec, getattr(dec, "_encoder", None)]:
                if obj is None:
                    continue
                for name in candidates:
                    if hasattr(obj, name):
                        pcm = _to_numpy(getattr(obj, name))
                        if pcm.ndim == 2 and pcm.size:
                            return (pcm.astype(np.uint8) & 1)
    except Exception:
        pass
    raise RuntimeError(
        "Could not find a parity-check matrix on Sionna LDPC5GEncoder. "
        "Known attributes were: " + ", ".join(sorted([a for a in dir(enc) if "pcm" in a.lower() or "pc" in a.lower()]))
    )


def _derive_5g_pruned_pcm_and_pattern(enc, requested_k: int, requested_n: int, pcm_full: np.ndarray):
    """Derive the internal, rate-matched 5G graph used by the custom BP/GRAND path.

    Sionna's 5G encoder returns the transmitted, rate-matched length-n codeword. The
    5G LDPC mother code punctures the first 2*z systematic positions. For GRAND/BP we
    use an internal vector consisting of:

        [k systematic bits] + [rate-matched parity bits]

    The transmitted vector is:

        internal[2*z:k] followed by internal[k:]

    Therefore internal_n = requested_n + 2*z and punctured positions are 0..2*z-1.
    This function removes shortened systematic columns and parity columns not present
    in the rate-matched output, and keeps the matching parity-check rows.
    """
    k = int(requested_k)
    n_tx = int(requested_n)
    z = int(_get_attr(enc, ["z", "_z"], 0) or 0)
    if z <= 0:
        # Fallback: infer from a common 5G relation. For k=256,bg2 this gives z=32.
        z = max(1, (int(_get_attr(enc, ["k_ldpc", "_k_ldpc"], k)) - k) // 2)
    k_ldpc = int(_get_attr(enc, ["k_ldpc", "_k_ldpc"], k) or k)
    n_ldpc = int(_get_attr(enc, ["n_ldpc", "_n_ldpc"], pcm_full.shape[1]) or pcm_full.shape[1])
    bg = str(_get_attr(enc, ["_bg", "bg"], ""))

    first_punctured = int(2 * z)
    if first_punctured >= k:
        raise RuntimeError(f"Invalid 5G LDPC parameters: first_punctured={first_punctured} >= k={k}")
    transmitted_systematic = k - first_punctured
    parity_keep = n_tx - transmitted_systematic
    if parity_keep <= 0:
        raise RuntimeError(
            f"requested n={n_tx} is too small for k={k}, z={z}: transmitted_systematic={transmitted_systematic}"
        )

    # If the PCM is already pruned to the desired internal length, use it directly.
    internal_n = k + parity_keep
    if pcm_full.shape[1] == internal_n and pcm_full.shape[0] == parity_keep:
        h = pcm_full.copy()
    else:
        # Mother-code column layout is [k_ldpc systematic incl. shortened] + parity.
        # Keep all real k systematic positions, remove shortened positions [k:k_ldpc),
        # and keep the first rate-matched parity_keep parity positions.
        if pcm_full.shape[1] < k_ldpc + parity_keep:
            raise RuntimeError(
                f"PCM has too few columns ({pcm_full.shape[1]}) for k_ldpc={k_ldpc}, parity_keep={parity_keep}"
            )
        keep_cols = np.concatenate([
            np.arange(0, k, dtype=np.int64),
            np.arange(k_ldpc, k_ldpc + parity_keep, dtype=np.int64),
        ])
        # For 5G LDPC, the first parity_keep check rows correspond to the kept parity section.
        if pcm_full.shape[0] < parity_keep:
            raise RuntimeError(f"PCM has too few rows ({pcm_full.shape[0]}) for parity_keep={parity_keep}")
        h = pcm_full[:parity_keep, :][:, keep_cols].copy()

    rm_pattern = np.ones(internal_n, dtype=np.int8)
    rm_pattern[:first_punctured] = 0
    meta = {
        "z": z,
        "k_ldpc": k_ldpc,
        "n_ldpc": n_ldpc,
        "bg": bg,
        "first_punctured": first_punctured,
        "parity_keep": parity_keep,
        "transmitted_systematic": transmitted_systematic,
    }
    return (h.astype(np.uint8) & 1), rm_pattern, meta


def build_sionna_nr_ldpc(
    k: int = 256,
    n: int = 512,
    align_to_pcm_length: bool = True,
    strict_pcm_check: bool = True,
    seed: int = 1234,
    **_,
) -> LDPCCode:
    enc = _instantiate_encoder(int(k), int(n))
    pcm_full = _extract_full_pcm(enc)
    h, rm_pattern, meta = _derive_5g_pruned_pcm_and_pattern(enc, int(k), int(n), pcm_full)
    k_int = int(k)
    n_int = int(h.shape[1])
    tx_pos = np.flatnonzero(rm_pattern == 1).astype(np.int32)
    punc_pos = np.flatnonzero(rm_pattern == 0).astype(np.int32)
    first_punctured = int(meta["first_punctured"])

    def _encode_internal(message: np.ndarray) -> np.ndarray:
        msg = np.asarray(message, dtype=np.uint8)
        squeeze = msg.ndim == 1
        if squeeze:
            msg = msg[None, :]
        if msg.shape[1] != k_int:
            raise ValueError(f"expected message length k={k_int}, got {msg.shape[1]}")
        # Sionna expects float bits and returns transmitted rate-matched bits.
        tx = enc(msg.astype(np.float32))
        tx = (_to_numpy(tx).reshape(msg.shape[0], -1) > 0.5).astype(np.uint8)
        if tx.shape[1] != int(n):
            raise RuntimeError(f"Sionna encoder returned length {tx.shape[1]}, expected transmitted n={n}")
        out = np.zeros((msg.shape[0], n_int), dtype=np.uint8)
        # The first k internal positions are systematic. This fixes the v10 no-op bug
        # that left punctured systematic positions at zero for nonzero messages.
        out[:, :k_int] = msg
        # Sionna's transmitted vector starts at internal position 2*z:
        # [systematic bits 2z..k-1] + [kept parity bits].
        sys_tx_len = k_int - first_punctured
        if sys_tx_len > 0:
            # Use the message for systematic positions to avoid any output interleaver ambiguity.
            out[:, first_punctured:k_int] = msg[:, first_punctured:k_int]
        parity = tx[:, sys_tx_len:]
        if parity.shape[1] != (n_int - k_int):
            raise RuntimeError(
                f"Sionna parity slice has {parity.shape[1]} bits, expected {n_int-k_int}; "
                f"tx_len={tx.shape[1]}, sys_tx_len={sys_tx_len}"
            )
        out[:, k_int:] = parity
        return out[0] if squeeze else out

    code = LDPCCode(
        h=h,
        k=k_int,
        n=n_int,
        family="sionna_nr_ldpc",
        rate=float(k) / float(n),
        g=None,
        encoder=_encode_internal,
        rm_pattern=rm_pattern,
        bg=meta.get("bg"),
        metadata=meta,
    )

    # Validate all-zero and random nonzero internal codeword reconstruction.
    if strict_pcm_check:
        rng = np.random.default_rng(seed)
        tests = [np.zeros(k_int, dtype=np.uint8)]
        tests += [rng.integers(0, 2, size=k_int, dtype=np.uint8) for _ in range(8)]
        bad = []
        for i, msg in enumerate(tests):
            c = code.encode_internal(msg)
            sw = int(code.syndrome(c).sum())
            if sw != 0:
                bad.append((i, sw))
        if bad:
            raise RuntimeError(
                "5G internal PCM/rate-matching validation failed on nonzero messages. "
                f"First failures: {bad[:5]}. This protects against silent all-zero-only alignment bugs."
            )
    return code
