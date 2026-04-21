from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def clip_llr(x: np.ndarray, clip: float = 18.0) -> np.ndarray:
    return np.clip(x, -clip, clip)


def bpsk_from_bits(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.astype(np.float32)
