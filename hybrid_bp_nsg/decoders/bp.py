from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..codes.peg_ldpc import LDPCCode
from ..utils.math_utils import clip_llr


@dataclass
class BPDecodeResult:
    success: bool
    hard: np.ndarray
    posterior_llr: np.ndarray
    syndrome: np.ndarray
    iterations_used: int
    elapsed_ms: float
    trace: Optional[Dict[str, np.ndarray]] = None


class BeliefPropagationDecoder:
    def __init__(self, code: LDPCCode, algorithm: str = "spa", max_iters: int = 20,
                 nms_alpha: float = 0.8, llr_clip: float = 18.0, early_stop: bool = True):
        self.code = code
        self.algorithm = algorithm.lower()
        self.max_iters = int(max_iters)
        self.nms_alpha = float(nms_alpha)
        self.llr_clip = float(llr_clip)
        self.early_stop = bool(early_stop)

    def decode(self, llr: np.ndarray, max_iters: Optional[int] = None, collect_trace: bool = False) -> BPDecodeResult:
        import time
        start = time.perf_counter()
        llr = clip_llr(np.asarray(llr, dtype=np.float32), self.llr_clip)
        max_iters = self.max_iters if max_iters is None else int(max_iters)
        E = self.code.edge_vars.size
        v2c = llr[self.code.edge_vars].astype(np.float32).copy()
        c2v = np.zeros(E, dtype=np.float32)
        posterior = llr.copy()
        hard = (posterior < 0).astype(np.uint8)
        syn = (self.code.h @ hard) % 2
        trace: Dict[str, List[np.ndarray]] = {
            "posterior_llr": [],
            "hard": [],
            "syndrome": [],
        } if collect_trace else {}
        if collect_trace:
            trace["posterior_llr"].append(posterior.copy())
            trace["hard"].append(hard.copy())
            trace["syndrome"].append(syn.copy())
        if syn.sum() == 0:
            elapsed_ms = (time.perf_counter() - start) * 1e3
            return BPDecodeResult(True, hard, posterior, syn, 0, elapsed_ms,
                                  self._stack_trace(trace) if collect_trace else None)

        for it in range(1, max_iters + 1):
            # Check update
            if self.algorithm == "spa":
                for c, edges in enumerate(self.code.check_edges):
                    msgs = np.clip(v2c[edges], -18.0, 18.0)
                    tanh_vals = np.tanh(msgs / 2.0)
                    abs_vals = np.clip(np.abs(tanh_vals), 1e-12, 1 - 1e-12)
                    sign_prod = np.prod(np.sign(tanh_vals))
                    prod_abs = np.prod(abs_vals)
                    for idx_local, e in enumerate(edges):
                        val = sign_prod * np.sign(tanh_vals[idx_local]) * (prod_abs / abs_vals[idx_local])
                        c2v[e] = 2.0 * np.arctanh(np.clip(val, -0.999999, 0.999999))
            elif self.algorithm in {"nms", "minsum", "normalized_minsum"}:
                for c, edges in enumerate(self.code.check_edges):
                    msgs = v2c[edges]
                    signs = np.sign(msgs)
                    signs[signs == 0] = 1.0
                    abs_msgs = np.abs(msgs)
                    order = np.argsort(abs_msgs)
                    min1 = abs_msgs[order[0]]
                    min2 = abs_msgs[order[1]] if abs_msgs.size > 1 else min1
                    sign_prod = np.prod(signs)
                    for idx_local, e in enumerate(edges):
                        mag = min2 if idx_local == order[0] else min1
                        c2v[e] = self.nms_alpha * sign_prod * signs[idx_local] * mag
            else:
                raise ValueError(f"Unsupported BP algorithm: {self.algorithm}")

            # Variable update / posterior
            for v, edges in enumerate(self.code.var_edges):
                total = llr[v] + np.sum(c2v[edges])
                posterior[v] = total
                for e in edges:
                    v2c[e] = total - c2v[e]

            hard = (posterior < 0).astype(np.uint8)
            syn = (self.code.h @ hard) % 2
            if collect_trace:
                trace["posterior_llr"].append(posterior.copy())
                trace["hard"].append(hard.copy())
                trace["syndrome"].append(syn.copy())
            if self.early_stop and syn.sum() == 0:
                elapsed_ms = (time.perf_counter() - start) * 1e3
                return BPDecodeResult(True, hard, posterior.copy(), syn.copy(), it, elapsed_ms,
                                      self._stack_trace(trace) if collect_trace else None)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        return BPDecodeResult(False, hard, posterior.copy(), syn.copy(), max_iters, elapsed_ms,
                              self._stack_trace(trace) if collect_trace else None)

    @staticmethod
    def _stack_trace(trace: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        if not trace:
            return {}
        out: Dict[str, np.ndarray] = {}
        for key, values in trace.items():
            out[key] = np.stack(values, axis=0)
        return out


class WeightedBitFlippingPostProcessor:
    def __init__(self, code: LDPCCode, max_steps: int = 10, damp: float = 0.2):
        self.code = code
        self.max_steps = int(max_steps)
        self.damp = float(damp)

    def decode(self, llr: np.ndarray, initial_hard: np.ndarray) -> BPDecodeResult:
        import time
        start = time.perf_counter()
        llr = np.asarray(llr, dtype=np.float32)
        hard = np.asarray(initial_hard, dtype=np.uint8).copy()
        for step in range(1, self.max_steps + 1):
            syn = (self.code.h @ hard) % 2
            if syn.sum() == 0:
                elapsed_ms = (time.perf_counter() - start) * 1e3
                posterior = llr * (1.0 - 2.0 * hard)
                return BPDecodeResult(True, hard.copy(), posterior, syn, step - 1, elapsed_ms)
            scores = np.zeros(self.code.n, dtype=np.float32)
            unsat_checks = np.flatnonzero(syn)
            if unsat_checks.size == 0:
                break
            for c in unsat_checks:
                vars_c = np.flatnonzero(self.code.h[c])
                reliab = np.maximum(np.min(np.abs(llr[vars_c])), 1e-3)
                scores[vars_c] += 1.0 / reliab
            scores -= self.damp * np.abs(llr)
            flip_idx = int(np.argmax(scores))
            hard[flip_idx] ^= 1
        syn = (self.code.h @ hard) % 2
        elapsed_ms = (time.perf_counter() - start) * 1e3
        posterior = llr * (1.0 - 2.0 * hard)
        return BPDecodeResult(bool(syn.sum() == 0), hard.copy(), posterior, syn, self.max_steps, elapsed_ms)
