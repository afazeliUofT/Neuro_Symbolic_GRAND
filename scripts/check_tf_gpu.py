#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
ap = argparse.ArgumentParser(); ap.add_argument('--require-gpu', action='store_true')
args = ap.parse_args()
import tensorflow as tf
print('tensorflow:', tf.__version__)
print('gpus:', tf.config.list_physical_devices('GPU'))
if args.require_gpu and not tf.config.list_physical_devices('GPU'):
    raise SystemExit('ERROR: no TensorFlow GPU visible')
