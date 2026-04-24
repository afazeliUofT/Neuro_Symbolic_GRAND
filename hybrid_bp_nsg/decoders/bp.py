from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..codes.peg_ldpc import LDPCCode


@dataclass
class BPDecodeResult:
    success: bool
    hard: np.ndarray
    posterior_llr: np.ndarray
    syndrome: np.ndarray
    iterations_used: int
    trace: Optional[Dict[str, np.ndarray]] = None
    crc_ok: bool = True


class BeliefPropagationDecoder:
    """Flooding normalized min-sum / sum-product LDPC decoder."""

    def __init__(
        self,
        code: LDPCCode,
        max_iters: int = 20,
        algorithm: str = "nms",
        nms_alpha: float = 0.8,
        early_stop: bool = True,
    ):
        self.code = code
        self.max_iters = int(max_iters)
        self.algorithm = str(algorithm).lower()
        self.nms_alpha = float(nms_alpha)
        self.early_stop = bool(early_stop)
        self._build_edges()

    def _build_edges(self) -> None:
        h = self.code.h
        rows, cols = np.nonzero(h)
        self.e_check = rows.astype(np.int32)
        self.e_var = cols.astype(np.int32)
        self.num_edges = int(len(rows))
        self.check_edges: List[np.ndarray] = [np.flatnonzero(self.e_check == c).astype(np.int32) for c in range(self.code.m)]
        self.var_edges: List[np.ndarray] = [np.flatnonzero(self.e_var == v).astype(np.int32) for v in range(self.code.n)]

    def _codeword_and_crc_ok(self, hard: np.ndarray) -> tuple[bool, bool]:
        syndrome = self.code.syndrome(hard)
        syn_ok = int(syndrome.sum()) == 0
        if not syn_ok:
            return False, False
        crc_ok = self.code.crc_check_internal(hard)
        return bool(syn_ok and crc_ok), bool(crc_ok)

    def decode(self, llr: np.ndarray, max_iters: Optional[int] = None, collect_trace: bool = False) -> BPDecodeResult:
        llr = np.asarray(llr, dtype=np.float32).reshape(-1)
        if llr.size != self.code.n:
            raise ValueError(f"expected internal LLR length {self.code.n}, got {llr.size}")
        max_iters = int(max_iters or self.max_iters)
        q = llr[self.e_var].astype(np.float32).copy()
        r = np.zeros_like(q, dtype=np.float32)
        posterior = llr.copy()
        hard = (posterior < 0).astype(np.uint8)
        syndrome = self.code.syndrome(hard)
        success, crc_ok = self._codeword_and_crc_ok(hard)

        trace_llr = []
        trace_hard = []
        trace_syn = []
        if collect_trace:
            trace_llr.append(posterior.copy())
            trace_hard.append(hard.copy())
            trace_syn.append(syndrome.copy())

        it_used = 0
        if success and self.early_stop:
            return BPDecodeResult(True, hard, posterior, syndrome, 0, {
                "posterior_llr": np.asarray(trace_llr, dtype=np.float32),
                "hard": np.asarray(trace_hard, dtype=np.uint8),
                "syndrome": np.asarray(trace_syn, dtype=np.uint8),
            } if collect_trace else None, crc_ok=crc_ok)

        for it in range(1, max_iters + 1):
            for edges in self.check_edges:
                if edges.size == 0:
                    continue
                vals = q[edges]
                signs = np.sign(vals)
                signs[signs == 0] = 1.0
                abs_vals = np.abs(vals)
                prod_sign = np.prod(signs)
                if edges.size == 1:
                    mins = np.array([0.0], dtype=np.float32)
                    out_sign = np.array([prod_sign], dtype=np.float32)
                else:
                    order = np.argsort(abs_vals)
                    min1 = abs_vals[order[0]]
                    min2 = abs_vals[order[1]]
                    mins = np.full(edges.size, min1, dtype=np.float32)
                    mins[order[0]] = min2
                    out_sign = prod_sign * signs
                if self.algorithm in {"nms", "normalized_min_sum", "normalized-min-sum"}:
                    r[edges] = self.nms_alpha * out_sign * mins
                elif self.algorithm in {"ms", "minsum", "min_sum"}:
                    r[edges] = out_sign * mins
                else:
                    t = np.tanh(np.clip(vals, -20.0, 20.0) / 2.0)
                    for j, e in enumerate(edges):
                        prod = np.prod(np.delete(t, j)) if edges.size > 1 else 1.0
                        prod = float(np.clip(prod, -0.999999, 0.999999))
                        r[e] = 2.0 * np.arctanh(prod)

            posterior = llr.copy()
            np.add.at(posterior, self.e_var, r)
            hard = (posterior < 0).astype(np.uint8)
            syndrome = self.code.syndrome(hard)
            success, crc_ok = self._codeword_and_crc_ok(hard)
            it_used = it

            if collect_trace:
                trace_llr.append(posterior.copy())
                trace_hard.append(hard.copy())
                trace_syn.append(syndrome.copy())

            q = posterior[self.e_var] - r

            if success and self.early_stop:
                break

        trace = None
        if collect_trace:
            trace = {
                "posterior_llr": np.asarray(trace_llr, dtype=np.float32),
                "hard": np.asarray(trace_hard, dtype=np.uint8),
                "syndrome": np.asarray(trace_syn, dtype=np.uint8),
            }
        return BPDecodeResult(bool(success), hard.astype(np.uint8), posterior.astype(np.float32), syndrome.astype(np.uint8), it_used, trace, crc_ok=crc_ok)
