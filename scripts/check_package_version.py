#!/usr/bin/env python3
from __future__ import annotations
import importlib.metadata as im
import os, sys
try:
    import hybrid_bp_nsg
    local = getattr(hybrid_bp_nsg, "__version__", "unknown")
except Exception as e:
    local = f"import-error:{e}"
try:
    dist = im.version("hybrid-bp-nsg-channel-aligned")
except Exception as e:
    dist = f"metadata-error:{e}"
print({
    "hybrid_bp_nsg.__version__": local,
    "distribution_version": dist,
    "python": sys.executable,
    "cwd": os.getcwd(),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
})
