from __future__ import annotations

from typing import Dict

from .peg_ldpc import build_peg_ldpc, LDPCCode


def build_code(cfg: Dict[str, object]) -> LDPCCode:
    family = str(cfg.get("family", "peg_ldpc")).lower()
    if family in {"peg_ldpc", "peg", "random_ldpc"}:
        return build_peg_ldpc(**cfg)
    if family in {"sionna_nr_ldpc", "nr5g", "5g", "5g_ldpc"}:
        from .sionna_nr_ldpc import build_sionna_nr_ldpc
        return build_sionna_nr_ldpc(**cfg)
    raise ValueError(f"Unknown code family: {family}")
