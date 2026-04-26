#!/usr/bin/env python3
from __future__ import annotations
import importlib, sys
mods = ["numpy", "yaml", "scipy", "tensorflow", "sionna"]
print("python:", sys.version)
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}:", getattr(mod, "__version__", "imported"))
    except Exception as e:
        print(f"{m}: MISSING/ERROR {type(e).__name__}: {e}")
