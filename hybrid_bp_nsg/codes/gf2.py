from __future__ import annotations

from typing import List, Tuple
import numpy as np


def gf2_rank(a: np.ndarray) -> int:
    a = np.array(a, dtype=np.uint8, copy=True)
    m, n = a.shape
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if a[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
        if r == m:
            break
    return r


def gf2_rref(a: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    a = np.array(a, dtype=np.uint8, copy=True)
    m, n = a.shape
    pivots: List[int] = []
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if a[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for i in range(m):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return a, pivots


def gf2_nullspace(a: np.ndarray) -> np.ndarray:
    rref, pivots = gf2_rref(a)
    m, n = rref.shape
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    basis = []
    for free in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        for i, p in enumerate(pivots):
            if rref[i, free]:
                vec[p] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8)


def h_to_generator(h: np.ndarray) -> np.ndarray:
    g = gf2_nullspace(h)
    if g.size == 0:
        raise ValueError("Nullspace is empty; H does not define a valid code")
    return g


def syndrome(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (h @ x.astype(np.uint8)) % 2
