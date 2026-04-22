from __future__ import annotations
import importlib, sys
mods = ["numpy", "yaml", "torch", "tensorflow", "sionna"]
ok = True
for name in mods:
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"{name}: MISSING/ERROR: {e}")
print("python:", sys.version.replace("\n", " "))
if not ok:
    raise SystemExit(2)
