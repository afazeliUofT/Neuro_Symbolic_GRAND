from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DecodeAttempt:
    stage: str
    query_index: int
    weight: int
    score: float
    positions: List[int]
    syndrome_weight: int
    success: bool


@dataclass
class DecodeResult:
    success: bool
    decoded_codeword: np.ndarray
    queries: int
    stage: str
    elapsed_ms: float
    fallback_used: bool
    candidate_pool_size: int
    oracle_pool_hit: Optional[bool] = None
    predicted_weight_top: Optional[int] = None
    confidence_prob: Optional[float] = None
    trace: List[DecodeAttempt] = field(default_factory=list)
