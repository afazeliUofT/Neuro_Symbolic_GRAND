from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def get_project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def detect_slurm_cpus(default: int = 1) -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        value = os.environ.get(key)
        if not value:
            continue
        if "(" in value:
            value = value.split("(", 1)[0]
        if "," in value:
            value = value.split(",", 1)[0]
        try:
            return max(1, int(value))
        except ValueError:
            continue
    return default


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_process_thread_env(num_threads: int) -> None:
    """Set conservative thread-related environment variables.

    This function should be called before importing heavy compute libraries
    such as TensorFlow in worker processes.
    """
    num_threads = max(1, int(num_threads))
    env_updates = {
        "OMP_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        "TF_NUM_INTRAOP_THREADS": str(num_threads),
        "TF_NUM_INTEROP_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    for key, value in env_updates.items():
        os.environ[key] = value


def configure_torch_threads(num_threads: int, num_interop_threads: int = 1) -> None:
    import torch

    num_threads = max(1, int(num_threads))
    num_interop_threads = max(1, int(num_interop_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        # set_num_interop_threads can only be called once in a process.
        pass


def configure_tensorflow_threads(num_intra_threads: int, num_inter_threads: int = 1) -> None:
    import tensorflow as tf

    num_intra_threads = max(1, int(num_intra_threads))
    num_inter_threads = max(1, int(num_inter_threads))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(num_intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_inter_threads)
    except RuntimeError:
        # TensorFlow can refuse once the runtime is initialized.
        pass


def runtime_snapshot(extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    snapshot: Dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "tf_num_intraop_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "tf_num_interop_threads": os.environ.get("TF_NUM_INTEROP_THREADS"),
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def write_runtime_snapshot(path: Path, extra: Optional[Dict[str, object]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(runtime_snapshot(extra=extra), f, indent=2, sort_keys=True)
