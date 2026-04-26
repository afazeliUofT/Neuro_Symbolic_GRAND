from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .code import LDPCCode


@dataclass
class BPDecodeResult:
    success: bool
    hard: np.ndarray
    posterior_llr: np.ndarray
    syndrome: np.ndarray
    iterations: int
    crc_ok: bool
    elapsed_ms: float = 0.0
    trace: Dict[str, np.ndarray] = field(default_factory=dict)


def bp_decode(code: LDPCCode, llr: np.ndarray, iterations: int = 20, nms_alpha: float = 0.8,
              early_stop: bool = True, collect_trace: bool = True) -> BPDecodeResult:
    start = time.perf_counter()
    llr = np.asarray(llr, dtype=np.float32).reshape(-1)
    m, n = code.m, code.n
    edges = code.edges
    e_count = edges.shape[0]
    cn_edges: List[np.ndarray] = code.cn_edge_indices
    vn_edges: List[np.ndarray] = code.vn_edge_indices

    v_to_c = np.zeros(e_count, dtype=np.float32)
    c_to_v = np.zeros(e_count, dtype=np.float32)
    for e, (_i, j) in enumerate(edges):
        v_to_c[e] = llr[j]

    hard_hist = []
    llr_hist = []
    syn_hist = []
    final_post = llr.copy()
    final_hard = (final_post < 0).astype(np.uint8)
    final_syn = code.syndrome(final_hard)

    for it in range(1, int(iterations) + 1):
        # Check-node update: normalized min-sum.
        for es in cn_edges:
            if es.size == 0:
                continue
            vals = v_to_c[es]
            signs = np.sign(vals)
            signs[signs == 0] = 1.0
            prod = float(np.prod(signs))
            absvals = np.abs(vals)
            if es.size == 1:
                c_to_v[es[0]] = 0.0
            else:
                min1_idx_local = int(np.argmin(absvals))
                min1 = float(absvals[min1_idx_local])
                tmp = absvals.copy()
                tmp[min1_idx_local] = np.inf
                min2 = float(np.min(tmp))
                for loc, e in enumerate(es):
                    mag = min2 if loc == min1_idx_local else min1
                    c_to_v[e] = float(nms_alpha) * prod * signs[loc] * mag
        # Variable-node update.
        final_post = llr.copy()
        for j, es in enumerate(vn_edges):
            if es.size:
                total = float(np.sum(c_to_v[es]))
                final_post[j] = llr[j] + total
                v_to_c[es] = final_post[j] - c_to_v[es]
        final_hard = (final_post < 0).astype(np.uint8)
        final_syn = code.syndrome(final_hard)
        if collect_trace:
            if it == 1 or it == iterations or it % 2 == 0:
                hard_hist.append(final_hard.copy())
                llr_hist.append(final_post.copy())
                syn_hist.append(final_syn.copy())
        crc_ok = code.crc_check_internal(final_hard)
        if early_stop and np.all(final_syn == 0) and crc_ok:
            break
    else:
        it = int(iterations)
        crc_ok = code.crc_check_internal(final_hard)

    success = bool(np.all(final_syn == 0) and crc_ok)
    trace: Dict[str, np.ndarray] = {}
    if collect_trace:
        if not hard_hist:
            hard_hist = [final_hard.copy()]
            llr_hist = [final_post.copy()]
            syn_hist = [final_syn.copy()]
        trace = {
            "hard": np.asarray(hard_hist, dtype=np.uint8),
            "posterior_llr": np.asarray(llr_hist, dtype=np.float32),
            "syndrome": np.asarray(syn_hist, dtype=np.uint8),
        }
    return BPDecodeResult(success=success, hard=final_hard.astype(np.uint8), posterior_llr=final_post.astype(np.float32),
                          syndrome=final_syn.astype(np.uint8), iterations=it, crc_ok=bool(crc_ok),
                          elapsed_ms=(time.perf_counter() - start) * 1e3, trace=trace)
