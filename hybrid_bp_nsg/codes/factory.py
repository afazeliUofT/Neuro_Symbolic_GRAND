from __future__ import annotations

from typing import Dict, Any

from .peg_ldpc import build_peg_ldpc, code_summary as peg_code_summary
from .sionna_nr_ldpc import build_sionna_nr_ldpc, code_summary as sionna_code_summary


def build_code(cfg: Dict[str, Any]):
    family = str(cfg.get("family", "sionna_nr_ldpc")).lower()
    if family in {"sionna_nr_ldpc", "nr5g", "5g_nr_ldpc", "sionna5g"}:
        return build_sionna_nr_ldpc(**cfg)
    if family in {"peg_ldpc", "custom_peg"}:
        allowed = {"n", "k", "variable_degree", "check_degree_hint", "seed", "peg_restarts"}
        return build_peg_ldpc(**{k: v for k, v in cfg.items() if k in allowed})
    raise ValueError(f"Unsupported code family: {family}")


def code_summary(code) -> Dict[str, int | float | str]:
    if getattr(code, "family", "") == "sionna_nr_ldpc":
        return sionna_code_summary(code)
    return peg_code_summary(code)
