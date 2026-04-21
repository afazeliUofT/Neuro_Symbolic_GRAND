from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .gf2 import gf2_rank, h_to_generator, syndrome


@dataclass
class LDPCCode:
    h: np.ndarray
    g: np.ndarray
    n: int
    k: int
    m: int
    variable_degree: int
    seed: int
    edge_vars: np.ndarray
    edge_checks: np.ndarray
    var_edges: List[np.ndarray]
    check_edges: List[np.ndarray]
    deg_v: np.ndarray
    deg_c: np.ndarray

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, message: np.ndarray) -> np.ndarray:
        message = np.asarray(message, dtype=np.uint8)
        if message.ndim == 1:
            if message.size != self.k:
                raise ValueError(f"Expected message length {self.k}, got {message.size}")
            return (message @ self.g) % 2
        if message.ndim == 2:
            if message.shape[1] != self.k:
                raise ValueError(f"Expected message shape (*,{self.k}), got {message.shape}")
            return (message @ self.g) % 2
        raise ValueError("message must be 1D or 2D")

    def syndrome(self, word: np.ndarray) -> np.ndarray:
        return syndrome(self.h, word)


def _next_checks_bfs(var_neighbors: List[List[int]], check_neighbors: List[List[int]], v: int) -> set[int]:
    reached_checks = set()
    frontier_vars = {v}
    frontier_checks = set()
    visited_vars = {v}
    visited_checks = set()
    while True:
        next_checks = set()
        for vv in frontier_vars:
            for cc in var_neighbors[vv]:
                if cc not in visited_checks:
                    next_checks.add(cc)
        if not next_checks:
            break
        frontier_checks = next_checks
        visited_checks |= next_checks
        reached_checks |= next_checks
        next_vars = set()
        for cc in frontier_checks:
            for vv in check_neighbors[cc]:
                if vv not in visited_vars:
                    next_vars.add(vv)
        if not next_vars:
            break
        frontier_vars = next_vars
        visited_vars |= next_vars
        if len(reached_checks) == len(check_neighbors):
            break
    return reached_checks


def build_peg_ldpc(n: int, k: int, variable_degree: int = 3, check_degree_hint: int = 6, seed: int = 31415,
                   peg_restarts: int = 20) -> LDPCCode:
    m = n - k
    rng = np.random.default_rng(seed)
    best_h = None
    best_score = None
    for restart in range(peg_restarts):
        var_neighbors: List[List[int]] = [[] for _ in range(n)]
        check_neighbors: List[List[int]] = [[] for _ in range(m)]
        check_deg = np.zeros(m, dtype=np.int32)
        for v in range(n):
            for e_idx in range(variable_degree):
                connected = set(var_neighbors[v])
                if e_idx == 0:
                    min_deg = int(check_deg.min())
                    candidates = [c for c in range(m) if check_deg[c] == min_deg and c not in connected]
                else:
                    reached = _next_checks_bfs(var_neighbors, check_neighbors, v)
                    candidates = [c for c in range(m) if c not in reached and c not in connected]
                    if not candidates:
                        min_deg = int(np.min([check_deg[c] for c in range(m) if c not in connected]))
                        candidates = [c for c in range(m) if c not in connected and check_deg[c] == min_deg]
                if not candidates:
                    candidates = [c for c in range(m) if c not in connected]
                degs = np.array([check_deg[c] for c in candidates])
                min_deg = degs.min()
                pool = [c for c in candidates if check_deg[c] == min_deg]
                chosen = int(rng.choice(pool))
                var_neighbors[v].append(chosen)
                check_neighbors[chosen].append(v)
                check_deg[chosen] += 1
        h = np.zeros((m, n), dtype=np.uint8)
        for v, checks in enumerate(var_neighbors):
            h[checks, v] = 1
        rank = gf2_rank(h)
        score = (rank, -int(np.var(check_deg)), -int(np.max(check_deg)), int(np.min(check_deg)))
        if best_score is None or score > best_score:
            best_score = score
            best_h = h.copy()
        if rank == m:
            break
    if best_h is None:
        raise RuntimeError("Failed to build LDPC parity-check matrix")
    h = best_h
    if gf2_rank(h) < m:
        raise RuntimeError("Constructed H is not full rank; increase peg_restarts or change seed")
    g = h_to_generator(h)
    if g.shape[0] != k:
        raise RuntimeError(f"Expected generator with {k} rows, got {g.shape[0]}")
    edge_vars = []
    edge_checks = []
    var_edges: List[List[int]] = [[] for _ in range(n)]
    check_edges: List[List[int]] = [[] for _ in range(m)]
    edge_idx = 0
    for c in range(m):
        for v in np.flatnonzero(h[c]):
            edge_vars.append(v)
            edge_checks.append(c)
            var_edges[v].append(edge_idx)
            check_edges[c].append(edge_idx)
            edge_idx += 1
    return LDPCCode(
        h=h,
        g=g,
        n=n,
        k=k,
        m=m,
        variable_degree=variable_degree,
        seed=seed,
        edge_vars=np.array(edge_vars, dtype=np.int32),
        edge_checks=np.array(edge_checks, dtype=np.int32),
        var_edges=[np.array(x, dtype=np.int32) for x in var_edges],
        check_edges=[np.array(x, dtype=np.int32) for x in check_edges],
        deg_v=np.array([len(x) for x in var_edges], dtype=np.int32),
        deg_c=np.array([len(x) for x in check_edges], dtype=np.int32),
    )


def code_summary(code: LDPCCode) -> Dict[str, int | float]:
    return {
        "n": code.n,
        "k": code.k,
        "m": code.m,
        "rate": code.rate,
        "edges": int(code.edge_vars.size),
        "avg_check_degree": float(code.deg_c.mean()),
        "max_check_degree": int(code.deg_c.max()),
        "min_check_degree": int(code.deg_c.min()),
    }
