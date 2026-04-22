from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..codes.peg_ldpc import LDPCCode
from .bp import BeliefPropagationDecoder, BPDecodeResult
from .grand_rescue import ResidualGrandRescueDecoder, RescueResult


@dataclass
class HybridDecodeResult:
    success: bool
    hard: np.ndarray
    codeword: np.ndarray
    main_success: bool
    rescue_invoked: bool
    rescue_success: bool
    queries: int
    action: str
    elapsed_ms: float
    used_micro_bp: bool
    main_result: BPDecodeResult


class HybridBPNSGDecoder:
    def __init__(self, code: LDPCCode, cfg: Dict[str, object], rescue_net=None, device: str = "cpu", mode: Optional[str] = None):
        bp_cfg = cfg["bp"]
        rescue_cfg = dict(cfg["rescue"])
        rescue_cfg["num_segments"] = int(cfg.get("model", {}).get("num_segments", rescue_cfg.get("num_segments", 8)))
        self.main_bp = BeliefPropagationDecoder(
            code,
            max_iters=int(bp_cfg.get("hybrid_main_iterations", 20)),
            algorithm=str(bp_cfg.get("hybrid_main_algorithm", "nms")),
            nms_alpha=float(bp_cfg.get("nms_alpha", 0.8)),
            early_stop=bool(bp_cfg.get("early_stop", True)),
        )
        self.micro_bp = BeliefPropagationDecoder(
            code,
            max_iters=int(bp_cfg.get("micro_iterations", 8)),
            algorithm=str(bp_cfg.get("hybrid_main_algorithm", "nms")),
            nms_alpha=float(bp_cfg.get("nms_alpha", 0.8)),
            early_stop=True,
        )
        self.rescue = ResidualGrandRescueDecoder(
            code, self.main_bp, self.micro_bp, rescue_cfg,
            mode=mode or str(rescue_cfg.get("mode", "ai")),
            rescue_net=rescue_net,
            device=device,
        )

    def decode(self, llr: np.ndarray, snr_db: float = 0.0, profile: str = "A", collect_trace: bool = True) -> HybridDecodeResult:
        import time
        t0 = time.perf_counter()
        main = self.main_bp.decode(llr, collect_trace=collect_trace)
        rr: RescueResult = self.rescue.rescue(llr, main, snr_db=snr_db, profile=profile)
        elapsed = (time.perf_counter() - t0) * 1e3
        return HybridDecodeResult(
            success=bool(rr.success),
            hard=rr.final_hard.copy(),
            codeword=rr.final_codeword.copy(),
            main_success=bool(main.success),
            rescue_invoked=bool(rr.rescue_invoked),
            rescue_success=bool(rr.success and rr.rescue_invoked),
            queries=int(rr.queries),
            action=str(rr.action),
            elapsed_ms=float(elapsed),
            used_micro_bp=bool(rr.used_micro_bp),
            main_result=main,
        )
