from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np


class NPZShardDataset:
    """Pure-NumPy NPZ dataset used by the TensorFlow v11.2 trainer.

    The old v11.1 dataset depended on torch.utils.data. FIR probes showed the
    working Sionna stack is TensorFlow/Sionna 1.2.2, so the dataset is now
    framework-neutral and can feed tf.data or eager TensorFlow loops.
    """

    def __init__(self, root: str | Path, preload: bool = True):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz shards found in {self.root}")
        self.lengths: List[int] = []
        for p in self.paths:
            with np.load(p, allow_pickle=False) as d:
                self.lengths.append(int(d["var_features"].shape[0]))
        self._arrays: Dict[str, np.ndarray] | None = None
        if preload:
            self._arrays = load_npz_directory(self.root)

    def __len__(self) -> int:
        return int(sum(self.lengths))

    @property
    def arrays(self) -> Dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = load_npz_directory(self.root)
        return self._arrays

    def iter_batches(self, batch_size: int, shuffle: bool = False, rng: np.random.Generator | None = None) -> Iterator[Dict[str, np.ndarray]]:
        arrays = self.arrays
        n = int(next(iter(arrays.values())).shape[0])
        idx = np.arange(n)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(idx)
        for lo in range(0, n, int(batch_size)):
            sub = idx[lo:lo + int(batch_size)]
            yield {k: v[sub] for k, v in arrays.items()}


def load_npz_directory(root: str | Path) -> Dict[str, np.ndarray]:
    root = Path(root)
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz shards found in {root}")
    keys: List[str] = []
    arrays_by_key: Dict[str, List[np.ndarray]] = {}
    for pi, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as d:
            if pi == 0:
                keys = list(d.files)
                arrays_by_key = {k: [] for k in keys}
            for k in keys:
                if k in d.files:
                    arr = d[k]
                    # Convert half precision to float32 for stable TensorFlow losses.
                    if arr.dtype == np.float16:
                        arr = arr.astype(np.float32)
                    arrays_by_key[k].append(arr)
    return {k: np.concatenate(v, axis=0) for k, v in arrays_by_key.items() if v}


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
            "num_candidates": int(d["candidate_features"].shape[1]) if "candidate_features" in d.files and d["candidate_features"].ndim >= 3 else 0,
        }
