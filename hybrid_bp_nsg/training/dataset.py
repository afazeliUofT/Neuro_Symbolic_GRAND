from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    class Dataset:  # type: ignore
        pass


class NPZShardDataset(Dataset):
    def __init__(self, root: str | Path):
        if torch is None:
            raise RuntimeError("PyTorch is required for NPZShardDataset.")
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz shards found in {self.root}")
        self.index: List[tuple[int, int]] = []
        self.lengths = []
        for pi, p in enumerate(self.paths):
            with np.load(p, allow_pickle=False) as d:
                n = int(d["var_features"].shape[0])
            self.lengths.append(n)
            self.index.extend((pi, i) for i in range(n))
        self._cache_pi = None
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _load(self, pi: int):
        if self._cache_pi != pi:
            self._cache = np.load(self.paths[pi], allow_pickle=False)
            self._cache_pi = pi
        return self._cache

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pi, ri = self.index[idx]
        d = self._load(pi)
        out = {}
        float_keys = ["var_features", "check_features", "global_features", "candidate_features"]
        long_keys = ["heuristic_order", "weight_label"]
        bin_keys = ["bit_labels", "segment_labels", "standard_reachable", "expanded_reachable", "rescueable", "candidate_labels", "candidate_valid"]
        for k in float_keys:
            if k in d.files:
                out[k] = torch.tensor(d[k][ri], dtype=torch.float32)
        for k in long_keys:
            if k in d.files:
                out[k] = torch.tensor(d[k][ri], dtype=torch.long)
        for k in bin_keys:
            if k in d.files:
                out[k] = torch.tensor(d[k][ri], dtype=torch.float32)
        return out


def infer_shapes(root: str | Path) -> Dict[str, int]:
    p = sorted(Path(root).glob("*.npz"))[0]
    with np.load(p, allow_pickle=False) as d:
        return {
            "num_var_features": int(d["var_features"].shape[-1]),
            "num_check_features": int(d["check_features"].shape[-1]),
            "num_global_features": int(d["global_features"].shape[-1]),
            "n": int(d["var_features"].shape[1]),
            "m": int(d["check_features"].shape[1]),
            "candidate_feature_dim": int(d["candidate_features"].shape[-1]) if "candidate_features" in d.files else 8,
        }
