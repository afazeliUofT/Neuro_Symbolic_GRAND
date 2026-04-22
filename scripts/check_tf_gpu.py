#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()

    print("=" * 88)
    print("TensorFlow GPU check")
    print("=" * 88)
    print("sys.executable:", sys.executable)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))
    print("SLURM_GPUS:", os.environ.get("SLURM_GPUS"))
    print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
    print("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS"))
    try:
        import tensorflow as tf
        print("tensorflow:", tf.__version__)
        try:
            print("build_info:", json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str))
        except Exception as e:
            print("build_info_failed:", repr(e))
        print("built_with_cuda:", tf.test.is_built_with_cuda())
        gpus = tf.config.list_physical_devices("GPU")
        print("physical_gpus:", gpus)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("memory_growth_failed:", gpu, repr(e))
            try:
                print("gpu_details:", tf.config.experimental.get_device_details(gpu))
            except Exception as e:
                print("gpu_details_failed:", repr(e))
        if args.require_gpu and not gpus:
            print("ERROR: --require-gpu was set but TensorFlow sees no physical GPU.", file=sys.stderr)
            return 2
        if gpus:
            with tf.device("/GPU:0"):
                a = tf.random.normal((2048, 2048))
                b = tf.linalg.matmul(a, a)
                print("gpu_matmul_shape:", b.shape, "mean:", float(tf.reduce_mean(tf.cast(b, tf.float32)).numpy()))
        else:
            a = tf.random.normal((256, 256))
            b = tf.linalg.matmul(a, a)
            print("cpu_matmul_shape:", b.shape, "mean:", float(tf.reduce_mean(b).numpy()))
        return 0
    except Exception as e:
        print("ERROR:", type(e).__name__, e, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
