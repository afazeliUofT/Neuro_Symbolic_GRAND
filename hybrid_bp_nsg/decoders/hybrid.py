from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch

from .bp import BeliefPropagationDecoder, WeightedBitFlippingPostProcessor, BPDecodeResult
from .grand_rescue import ResidualGrandRescueDecoder, RescueResult
from ..models.rescue_net import RescueNet


@dataclass
class DecoderRunResult:
    decoder_name: str
    success: bool
    final_hard: np.ndarray
    block_error: int
    latency_ms: float
    queries: int
    action: str
    rescue_used: int
    micro_bp_used: int
    main_success: int


class HybridDecoderFactory:
    def __init__(self, code: object, cfg: Dict[str, object], checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.code = code
        self.cfg = cfg
        self.device = device
        bp_cfg = cfg["bp"]
        self.bp_10 = BeliefPropagationDecoder(code, algorithm=str(bp_cfg["algorithm"]), max_iters=10,
                                              nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=bool(bp_cfg["early_stop"]))
        self.bp_20 = BeliefPropagationDecoder(code, algorithm=str(bp_cfg["algorithm"]), max_iters=int(bp_cfg["main_iterations"]),
                                              nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=bool(bp_cfg["early_stop"]))
        self.bp_50 = BeliefPropagationDecoder(code, algorithm=str(bp_cfg["algorithm"]), max_iters=int(bp_cfg["strong_iterations"]),
                                              nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=bool(bp_cfg["early_stop"]))
        self.hybrid_main_bp = BeliefPropagationDecoder(code, algorithm=str(bp_cfg.get("hybrid_main_algorithm", "nms")), max_iters=int(bp_cfg.get("hybrid_main_iterations", 20)),
                                                       nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=bool(bp_cfg["early_stop"]))
        self.hybrid_micro_bp = BeliefPropagationDecoder(code, algorithm=str(bp_cfg.get("hybrid_micro_algorithm", bp_cfg.get("hybrid_main_algorithm", "nms"))), max_iters=int(bp_cfg["micro_iterations"]),
                                                        nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=bool(bp_cfg["early_stop"]))
        self.nms_20 = BeliefPropagationDecoder(code, algorithm="nms", max_iters=20, nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=True)
        self.nms_50 = BeliefPropagationDecoder(code, algorithm="nms", max_iters=50, nms_alpha=float(bp_cfg["nms_alpha"]), llr_clip=float(bp_cfg["llr_clip"]), early_stop=True)
        self.wbf = WeightedBitFlippingPostProcessor(code, max_steps=10)
        self.rescue_net = None
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            train_cfg = ckpt.get("cfg", cfg)
            model = RescueNet(
                num_var_features=12,
                num_check_features=5,
                num_global_features=8,
                n=code.n,
                m=code.m,
                num_segments=int(train_cfg["model"]["num_segments"]),
                max_weight_class=int(train_cfg["model"]["max_weight_class"]),
                hidden_dim=int(train_cfg["model"]["graph_hidden_dim"]),
                graph_layers=int(train_cfg["model"]["graph_layers"]),
                top_k_tokens=int(train_cfg["model"]["top_k_tokens"]),
                transformer_heads=int(train_cfg["model"]["transformer_heads"]),
                transformer_layers=int(train_cfg["model"]["transformer_layers"]),
                dropout=float(train_cfg["train"]["dropout"]),
                candidate_feature_dim=8,
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.rescue_net = model
        self.rescue_decoders = {
            "bp_orb_rescue": ResidualGrandRescueDecoder(code, self.hybrid_main_bp, self.hybrid_micro_bp, cfg["rescue"], mode="orb", device=device),
            "bp_cdf_rescue": ResidualGrandRescueDecoder(code, self.hybrid_main_bp, self.hybrid_micro_bp, cfg["rescue"], mode="cdf", device=device),
            "bp_segmented_rescue": ResidualGrandRescueDecoder(code, self.hybrid_main_bp, self.hybrid_micro_bp, cfg["rescue"], mode="segmented", device=device),
            "hybrid_bp_nsg": ResidualGrandRescueDecoder(code, self.hybrid_main_bp, self.hybrid_micro_bp, cfg["rescue"], mode="ai", rescue_net=self.rescue_net, device=device),
        }


    def evaluate_all(self, llr: np.ndarray, true_codeword: np.ndarray, snr_db: float, profile: str) -> Dict[str, DecoderRunResult]:
        results: Dict[str, DecoderRunResult] = {}
        bp10 = self.bp_10.decode(llr)
        results["bp_10"] = self._wrap_bp("bp_10", bp10, true_codeword)
        spa20 = self.bp_20.decode(llr)
        results["bp_20"] = self._wrap_bp("bp_20", spa20, true_codeword)
        spa50 = self.bp_50.decode(llr)
        results["bp_50"] = self._wrap_bp("bp_50", spa50, true_codeword)
        main = self.hybrid_main_bp.decode(llr, collect_trace=True)
        nms20 = self.nms_20.decode(llr)
        results["bp_nms_20"] = self._wrap_bp("bp_nms_20", nms20, true_codeword)
        nms50 = self.nms_50.decode(llr)
        results["bp_nms_50"] = self._wrap_bp("bp_nms_50", nms50, true_codeword)
        if main.success:
            results["bp_wbf_post"] = DecoderRunResult("bp_wbf_post", True, main.hard.copy(), int(np.any(main.hard[:self.code.k] != true_codeword[:self.code.k])), main.elapsed_ms, 0, "bp_success", 0, 0, 1)
            for name in self.rescue_decoders:
                results[name] = DecoderRunResult(name, True, main.hard.copy(), int(np.any(main.hard[:self.code.k] != true_codeword[:self.code.k])), main.elapsed_ms, 0, "bp_success", 0, 0, 1)
        else:
            post = self.wbf.decode(llr, main.hard)
            hard = post.hard if post.success else main.hard
            results["bp_wbf_post"] = DecoderRunResult("bp_wbf_post", bool(post.success), hard, int(np.any(hard[:self.code.k] != true_codeword[:self.code.k])), main.elapsed_ms + post.elapsed_ms, int(post.iterations_used), "wbf_success" if post.success else "wbf_fail", 1, 0, 0)
            for name, dec in self.rescue_decoders.items():
                res = dec.rescue(llr, main, snr_db, profile)
                results[name] = DecoderRunResult(name, res.success, res.final_hard, int(np.any(res.final_hard[:self.code.k] != true_codeword[:self.code.k])), main.elapsed_ms + res.elapsed_ms, res.queries, res.action, int(res.rescue_invoked), int(res.used_micro_bp), int(main.success))
        return results

    def decode(self, decoder_name: str, llr: np.ndarray, true_codeword: np.ndarray, snr_db: float, profile: str) -> DecoderRunResult:
        if decoder_name == "bp_10":
            res = BeliefPropagationDecoder(self.code, algorithm=self.bp_20.algorithm, max_iters=10, nms_alpha=self.bp_20.nms_alpha, llr_clip=self.bp_20.llr_clip).decode(llr)
            return self._wrap_bp(decoder_name, res, true_codeword)
        if decoder_name == "bp_20":
            res = self.bp_20.decode(llr)
            return self._wrap_bp(decoder_name, res, true_codeword)
        if decoder_name == "bp_50":
            res = self.bp_50.decode(llr)
            return self._wrap_bp(decoder_name, res, true_codeword)
        if decoder_name == "bp_nms_20":
            res = self.nms_20.decode(llr)
            return self._wrap_bp(decoder_name, res, true_codeword)
        if decoder_name == "bp_nms_50":
            res = self.nms_50.decode(llr)
            return self._wrap_bp(decoder_name, res, true_codeword)
        if decoder_name == "bp_wbf_post":
            main = self.hybrid_main_bp.decode(llr)
            if main.success:
                return self._wrap_bp(decoder_name, main, true_codeword)
            post = self.wbf.decode(llr, main.hard)
            latency = main.elapsed_ms + post.elapsed_ms
            hard = post.hard if post.success else main.hard
            return DecoderRunResult(decoder_name, bool(post.success), hard, int(np.any(hard[:self.code.k] != true_codeword[:self.code.k])), latency, int(post.iterations_used),
                                    "wbf_success" if post.success else "wbf_fail", 1, 0, int(main.success))
        if decoder_name in self.rescue_decoders:
            res = self.rescue_decoders[decoder_name].rescue(llr, self.hybrid_main_bp.decode(llr, collect_trace=True), snr_db, profile)
            latency = res.main_result.elapsed_ms + res.elapsed_ms
            return DecoderRunResult(decoder_name, res.success, res.final_hard, int(np.any(res.final_hard[:self.code.k] != true_codeword[:self.code.k])), latency,
                                    res.queries, res.action, int(res.rescue_invoked), int(res.used_micro_bp), int(res.main_result.success))
        raise ValueError(f"Unknown decoder {decoder_name}")


    def _wrap_bp(self, decoder_name: str, res: BPDecodeResult, true_codeword: np.ndarray) -> DecoderRunResult:
        return DecoderRunResult(decoder_name, res.success, res.hard, int(np.any(res.hard[:self.code.k] != true_codeword[:self.code.k])), res.elapsed_ms,
                                int(res.iterations_used), "bp_success" if res.success else "bp_fail", 0, 0, int(res.success))
