from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def gf2_rank(a: np.ndarray) -> int:
    a = (np.asarray(a, dtype=np.uint8) & 1).copy()
    m, n = a.shape
    r = 0
    for c in range(n):
        piv = np.flatnonzero(a[r:, c])
        if piv.size == 0:
            continue
        p = int(piv[0] + r)
        if p != r:
            a[[r, p]] = a[[p, r]]
        rows = np.flatnonzero(a[:, c])
        rows = rows[rows != r]
        if rows.size:
            a[rows] ^= a[r]
        r += 1
        if r == m:
            break
    return int(r)


def gf2_rref(a: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    a = (np.asarray(a, dtype=np.uint8) & 1).copy()
    m, n = a.shape
    pivots: List[int] = []
    r = 0
    for c in range(n):
        piv = np.flatnonzero(a[r:, c])
        if piv.size == 0:
            continue
        p = int(piv[0] + r)
        if p != r:
            a[[r, p]] = a[[p, r]]
        rows = np.flatnonzero(a[:, c])
        rows = rows[rows != r]
        if rows.size:
            a[rows] ^= a[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return a, pivots


def gf2_nullspace(h: np.ndarray) -> np.ndarray:
    """Return a basis G whose rows span the nullspace of H over GF(2)."""
    h = (np.asarray(h, dtype=np.uint8) & 1)
    m, n = h.shape
    rref, pivots = gf2_rref(h)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8)
    rows = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for row, p in enumerate(pivots):
            if rref[row, f]:
                x[p] = 1
        rows.append(x)
    return np.asarray(rows, dtype=np.uint8)


def gf2_solve(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solve A x = b over GF(2). Returns one solution with free vars set to zero."""
    a = (np.asarray(a, dtype=np.uint8) & 1)
    b = (np.asarray(b, dtype=np.uint8).reshape(-1, 1) & 1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"shape mismatch: A has {a.shape[0]} rows but b has {b.shape[0]}")
    aug = np.concatenate([a.copy(), b.copy()], axis=1)
    m, n1 = aug.shape
    n = n1 - 1
    pivots: List[int] = []
    r = 0
    for c in range(n):
        piv = np.flatnonzero(aug[r:, c])
        if piv.size == 0:
            continue
        p = int(piv[0] + r)
        if p != r:
            aug[[r, p]] = aug[[p, r]]
        rows = np.flatnonzero(aug[:, c])
        rows = rows[rows != r]
        if rows.size:
            aug[rows] ^= aug[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    # Inconsistency check: zero coefficients with RHS 1.
    if r < m:
        zero_coeff = np.sum(aug[r:, :n], axis=1) == 0
        bad = zero_coeff & (aug[r:, n] == 1)
        if np.any(bad):
            return None
    x = np.zeros(n, dtype=np.uint8)
    for row, col in enumerate(pivots):
        x[col] = aug[row, n]
    return x


def gf2_solve_support(h: np.ndarray, support: np.ndarray, syndrome: np.ndarray) -> Optional[np.ndarray]:
    support = np.asarray(support, dtype=np.int64)
    if support.size == 0:
        return np.zeros(0, dtype=np.uint8) if np.sum(syndrome) == 0 else None
    sub = (np.asarray(h, dtype=np.uint8)[:, support] & 1)
    return gf2_solve(sub, syndrome)
