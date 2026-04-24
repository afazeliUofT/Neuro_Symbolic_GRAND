from __future__ import annotations

import os
from typing import Any, Tuple


def _set_thread_env(num_threads: int | None) -> None:
    if num_threads is None or int(num_threads) <= 0:
        return
    n = str(int(num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", n)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", max(1, int(num_threads) // 4).__str__())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)


def configure_tensorflow(
    require_gpu: bool = False,
    mixed_precision: bool = False,
    xla: bool = True,
    cpu_threads: int | None = None,
    log_prefix: str = "tf",
):
    """Import and configure TensorFlow for FIR jobs.

    The configuration is intentionally done before model construction so that
    TensorFlow can enable memory growth, optional XLA graph compilation, and a
    bounded CPU thread pool that plays well with 32-core Slurm allocations.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    _set_thread_env(cpu_threads)

    import tensorflow as tf  # type: ignore

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if cpu_threads and int(cpu_threads) > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(cpu_threads))
            tf.config.threading.set_inter_op_parallelism_threads(max(1, int(cpu_threads) // 4))
        except Exception:
            pass

    if xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass
        try:
            tf.config.optimizer.set_experimental_options({
                "layout_optimizer": True,
                "constant_folding": True,
                "shape_optimization": True,
                "remapping": True,
                "arithmetic_optimization": True,
            })
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
    else:
        try:
            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:
            pass
    return tf, gpus


def tensorflow_device_string() -> str:
    try:
        import tensorflow as tf  # type: ignore
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    except Exception:
        return "/CPU:0"
