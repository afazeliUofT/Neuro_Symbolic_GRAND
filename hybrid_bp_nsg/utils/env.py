from __future__ import annotations

import os


def configure_tensorflow_threads(intra_threads: int = 1, inter_threads: int = 1) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
    except Exception:
        pass
