from __future__ import annotations

import os
from typing import Any


def _set_thread_env(num_threads: int | None) -> None:
    if num_threads is None or int(num_threads) <= 0:
        return
    n = str(int(num_threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["TF_NUM_INTRAOP_THREADS"] = n
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, int(num_threads) // 4))
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


def _normalize_policy(mixed_precision: Any, has_gpu: bool) -> str:
    if not has_gpu:
        return "float32"
    if isinstance(mixed_precision, str):
        key = mixed_precision.strip().lower()
        if key in {"", "none", "off", "false", "0", "float32", "fp32"}:
            return "float32"
        if key in {"true", "1", "auto", "yes", "bf16", "bfloat16", "mixed_bfloat16"}:
            return "mixed_bfloat16"
        if key in {"fp16", "float16", "half", "mixed_float16"}:
            return "mixed_float16"
        return "mixed_bfloat16"
    return "mixed_bfloat16" if bool(mixed_precision) else "float32"


def configure_tensorflow(
    require_gpu: bool = False,
    mixed_precision: Any = False,
    xla: bool = True,
    cpu_threads: int | None = None,
    log_prefix: str = "tf",
):
    """Import and configure TensorFlow for FIR jobs.

    Defaults are chosen for FIR H100 jobs:
      * TF32 enabled for float32 math on GPU
      * optional XLA graph compilation
      * mixed_bfloat16 by default when mixed_precision is requested
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    _set_thread_env(cpu_threads)

    import tensorflow as tf  # type: ignore

    try:
        tf.keras.backend.set_floatx("float32")
    except Exception:
        pass

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

    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
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

    policy = _normalize_policy(mixed_precision, has_gpu=bool(gpus))
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
    except Exception:
        tf.keras.mixed_precision.set_global_policy("float32")
    return tf, gpus


def tensorflow_device_string() -> str:
    try:
        import tensorflow as tf  # type: ignore
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    except Exception:
        return "/CPU:0"
