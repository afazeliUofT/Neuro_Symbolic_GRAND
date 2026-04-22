from __future__ import annotations

import os
from typing import Any


def configure_tensorflow(require_gpu: bool = False, mixed_precision: bool = False, log_prefix: str = "tf"):
    """Import and configure TensorFlow for FIR GPU jobs.

    Must be called before building models. It sets memory growth to avoid TF
    pre-allocating all H100/MIG memory and optionally enables mixed precision.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf  # type: ignore

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if require_gpu and not gpus:
        raise RuntimeError(
            "TensorFlow sees no GPU. On FIR submit with an explicit GPU type, e.g. "
            "`#SBATCH --gpus-per-node=h100:1` or one of the listed MIG H100 types."
        )

    if mixed_precision and gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    return tf, gpus


def tensorflow_device_string() -> str:
    try:
        import tensorflow as tf  # type: ignore
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    except Exception:
        return "/CPU:0"
