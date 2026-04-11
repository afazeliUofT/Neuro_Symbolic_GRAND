from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@lru_cache(maxsize=128)
def cached_combinations(pool_size: int, max_weight: int) -> Tuple[Tuple[int, ...], ...]:
    """Return candidate local-index tuples sorted by GRAND-like rank cost.

    The tuples are sorted by increasing sum of local ranks, then by weight,
    then lexicographically. This mirrors a simple ORBGRAND-like ordering on a
    reliability-sorted candidate pool.
    """
    combos: List[Tuple[int, ...]] = [tuple()]
    for weight in range(1, max_weight + 1):
        combos.extend(tuple(c) for c in combinations(range(pool_size), weight))
    combos.sort(key=lambda c: (sum(c), len(c), c))
    return tuple(combos)


def score_combinations(
    pool_scores: np.ndarray,
    max_weight: int,
    allowed_weights: Iterable[int] | None = None,
    weight_penalties: Sequence[float] | None = None,
    top_k: int | None = None,
) -> List[Tuple[float, Tuple[int, ...]]]:
    pool_scores = np.asarray(pool_scores, dtype=np.float64)
    combos = cached_combinations(len(pool_scores), max_weight)
    allowed_set = set(range(max_weight + 1)) if allowed_weights is None else set(int(w) for w in allowed_weights)
    scored: List[Tuple[float, Tuple[int, ...]]] = []
    for combo in combos:
        weight = len(combo)
        if weight not in allowed_set:
            continue
        penalty = 0.0 if weight_penalties is None else float(weight_penalties[min(weight, len(weight_penalties) - 1)])
        score = penalty + float(pool_scores[list(combo)].sum()) if combo else penalty
        scored.append((score, combo))
    scored.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    if top_k is not None:
        return scored[:top_k]
    return scored
