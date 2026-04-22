from __future__ import annotations

import numpy as np


def ebn0_db_to_noise_var(ebn0_db: float, rate: float = 1.0) -> float:
    ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
    return 1.0 / max(1e-12, 2.0 * rate * ebn0)


def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
