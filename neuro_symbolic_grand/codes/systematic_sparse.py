from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from ..utils.io import read_json, write_json


@dataclass
class SystematicSparseCode:
    n: int
    k: int
    p_matrix: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        self.p_matrix = np.asarray(self.p_matrix, dtype=np.uint8) % 2
        self.m = self.n - self.k
        if self.p_matrix.shape != (self.k, self.m):
            raise ValueError(f"Expected P matrix shape {(self.k, self.m)}, got {self.p_matrix.shape}")
        self.generator = np.concatenate([np.eye(self.k, dtype=np.uint8), self.p_matrix], axis=1)
        self.parity_check = np.concatenate([self.p_matrix.T, np.eye(self.m, dtype=np.uint8)], axis=1)
        self.column_weights = self.parity_check.sum(axis=0).astype(np.int64)
        self.row_weights = self.parity_check.sum(axis=1).astype(np.int64)
        self.rate = self.k / self.n

    def encode(self, messages: np.ndarray) -> np.ndarray:
        messages = np.asarray(messages, dtype=np.uint8) % 2
        if messages.ndim == 1:
            messages = messages[None, :]
        if messages.shape[1] != self.k:
            raise ValueError(f"Expected message width {self.k}, got {messages.shape[1]}")
        parity = (messages @ self.p_matrix) % 2
        return np.concatenate([messages, parity], axis=1).astype(np.uint8)

    def syndrome(self, words: np.ndarray) -> np.ndarray:
        words = np.asarray(words, dtype=np.uint8) % 2
        if words.ndim == 1:
            words = words[None, :]
        return (words @ self.parity_check.T) % 2

    def is_codeword(self, word: np.ndarray) -> bool:
        return bool(np.all(self.syndrome(word) == 0))

    def hard_error_mask(self, hard_bits: np.ndarray, codewords: np.ndarray) -> np.ndarray:
        return (np.asarray(hard_bits, dtype=np.uint8) ^ np.asarray(codewords, dtype=np.uint8)).astype(np.uint8)

    def unsatisfied_check_counts(self, word: np.ndarray) -> np.ndarray:
        word = np.asarray(word, dtype=np.uint8) % 2
        if word.ndim != 1:
            raise ValueError("unsatisfied_check_counts expects a single codeword")
        syndrome = self.syndrome(word)[0].astype(np.int64)
        return (self.parity_check.T @ syndrome).astype(np.int64)

    def to_artifact_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "k": self.k,
            "seed": self.seed,
            "p_matrix": self.p_matrix.tolist(),
        }

    def save(self, path: Path) -> None:
        write_json(self.to_artifact_dict(), path)


def build_systematic_sparse_code(n: int, k: int, p_column_weight: int, seed: int) -> SystematicSparseCode:
    if not 0 < k < n:
        raise ValueError("Require 0 < k < n")
    m = n - k
    p_column_weight = max(1, min(int(p_column_weight), k))
    rng = np.random.default_rng(seed)
    p_matrix = np.zeros((k, m), dtype=np.uint8)
    target_row_load = np.zeros(k, dtype=np.int64)
    for col in range(m):
        probs = 1.0 / (1.0 + target_row_load.astype(np.float64))
        probs = probs / probs.sum()
        chosen = rng.choice(k, size=p_column_weight, replace=False, p=probs)
        p_matrix[chosen, col] = 1
        target_row_load[chosen] += 1
    # Force every information bit to appear in at least one parity equation.
    isolated_rows = np.where(p_matrix.sum(axis=1) == 0)[0]
    for row in isolated_rows:
        col = int(rng.integers(0, m))
        p_matrix[row, col] = 1
    return SystematicSparseCode(n=n, k=k, p_matrix=p_matrix, seed=seed)


def load_code_artifact(path: Path) -> SystematicSparseCode:
    raw = read_json(path)
    return SystematicSparseCode(
        n=int(raw["n"]),
        k=int(raw["k"]),
        p_matrix=np.asarray(raw["p_matrix"], dtype=np.uint8),
        seed=int(raw.get("seed", 0)),
    )
