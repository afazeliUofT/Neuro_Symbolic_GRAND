from __future__ import annotations
import importlib
import sys

required = ["numpy", "yaml", "tensorflow", "sionna"]
optional = ["torch"]
ok = True
for name in required:
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"{name}: MISSING/ERROR: {e}")
for name in optional:
    try:
        m = importlib.import_module(name)
        print(f"{name}: optional OK {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"{name}: optional not importable ({e})")
print("python:", sys.version.replace("\n", " "))
if not ok:
    raise SystemExit(2)
